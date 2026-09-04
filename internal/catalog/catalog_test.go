// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package catalog

import (
	"strings"
	"testing"
)

const gb = 1 << 30

func dev(name string, g float64, unified bool) Device {
	return Device{Name: name, TotalBytes: uint64(g * gb), Unified: unified}
}

func entry(id string, total, perDev uint64, n int) Entry {
	return Entry{
		ID: id, Model: "m", Quant: "q",
		MinTotalBytes: total, MinPerDeviceBytes: perDev, MinDevices: n,
		Streams: 48, TotalContext: 425984,
		Provenance: Measured,
	}
}

func TestBuiltinCatalogParses(t *testing.T) {
	c, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Entries) == 0 {
		t.Fatal("the embedded catalog is empty")
	}
	for _, e := range c.Entries {
		if e.Provenance == Measured && e.Note == "" {
			t.Errorf("%s is measured but records nothing about the measurement", e.ID)
		}
	}
}

// ⛔ The rule the whole package exists for.
func TestComputedEntriesAreNeverInstallable(t *testing.T) {
	e := entry("candidate", 8*gb, 8*gb, 1)
	e.Provenance = Computed
	c := Catalog{Entries: []Entry{e}}
	ok, no := c.Installable([]Device{dev("huge", 80, false)}, DefaultMargins())
	if len(ok) != 0 {
		t.Fatal("a computed entry fits the arithmetic and has never been run; 128 streams fit the arithmetic too and killed the container")
	}
	if !strings.Contains(no[0].Reason, "never been run") {
		t.Fatalf("the reason must say why: %q", no[0].Reason)
	}
}

func TestFitsOnASufficientCard(t *testing.T) {
	c := Catalog{Entries: []Entry{entry("a", 20*gb, 20*gb, 1)}}
	ok, no := c.Installable([]Device{dev("3090", 24, false)}, DefaultMargins())
	if len(ok) != 1 || len(no) != 0 {
		t.Fatalf("24 GiB minus 5%% margin is 22.8, which clears 20: ok=%d no=%v", len(ok), no)
	}
}

func TestMarginIsApplied(t *testing.T) {
	// 24 GiB with a 5% margin leaves 22.8 — an entry needing 23.5 must not fit.
	c := Catalog{Entries: []Entry{entry("tight", uint64(23.5*gb), uint64(23.5*gb), 1)}}
	ok, _ := c.Installable([]Device{dev("3090", 24, false)}, DefaultMargins())
	if len(ok) != 0 {
		t.Fatal("an entry that fits only by consuming every byte will fail on a machine doing anything else")
	}
}

// The Apple Silicon asymmetry.
func TestUnifiedMemoryGetsALargerMargin(t *testing.T) {
	e := entry("mid", uint64(40*gb), uint64(40*gb), 1)
	c := Catalog{Entries: []Entry{e}}

	// 48 GiB dedicated: 5% margin leaves 45.6 -> fits.
	if ok, _ := c.Installable([]Device{dev("card", 48, false)}, DefaultMargins()); len(ok) != 1 {
		t.Fatal("48 GiB of dedicated memory should hold a 40 GiB model")
	}
	// 48 GiB unified: 20% margin leaves 38.4 -> does not fit, because that ceiling is shared
	// with everything else the machine is doing.
	if ok, no := c.Installable([]Device{dev("M-series", 48, true)}, DefaultMargins()); len(ok) != 0 {
		t.Fatalf("unified memory is shared with the host and needs more headroom: %v %v", ok, no)
	}
}

// The constraint people are surprised by.
func TestManySmallCardsDoNotSubstituteForFewLargeOnes(t *testing.T) {
	// 40 GiB total, but each participating device must hold 20.
	c := Catalog{Entries: []Entry{entry("big", 40*gb, 20*gb, 2)}}
	six := []Device{}
	for i := 0; i < 6; i++ {
		six = append(six, dev("small", 12, false))
	}
	ok, no := c.Installable(six, DefaultMargins())
	if len(ok) != 0 {
		t.Fatal("72 GiB of total memory across 12 GiB cards cannot hold a layer that needs 20")
	}
	if !strings.Contains(no[0].Reason, "each of") {
		t.Fatalf("the reason must name the per-device floor, not the total: %q", no[0].Reason)
	}
}

func TestTwoLargeCardsDoSubstitute(t *testing.T) {
	c := Catalog{Entries: []Entry{entry("big", 40*gb, 20*gb, 2)}}
	two := []Device{dev("3090", 24, false), dev("3090", 24, false)}
	if ok, no := c.Installable(two, DefaultMargins()); len(ok) != 1 {
		t.Fatalf("2x24 clears both the total and the per-device floor: %v", no)
	}
}

func TestLargestDevicesAreMatchedFirst(t *testing.T) {
	c := Catalog{Entries: []Entry{entry("big", 40*gb, 20*gb, 2)}}
	mixed := []Device{dev("small", 8, false), dev("3090", 24, false), dev("4090", 24, false)}
	if ok, no := c.Installable(mixed, DefaultMargins()); len(ok) != 1 {
		t.Fatalf("a two-device model should be matched against the two biggest, not the first two: %v", no)
	}
}

func TestNotEnoughDevices(t *testing.T) {
	c := Catalog{Entries: []Entry{entry("big", 40*gb, 20*gb, 2)}}
	_, no := c.Installable([]Device{dev("3090", 24, false)}, DefaultMargins())
	if !strings.Contains(no[0].Reason, "needs 2 devices, found 1") {
		t.Fatalf("got %q", no[0].Reason)
	}
}

func TestNoDevices(t *testing.T) {
	c := Catalog{Entries: []Entry{entry("a", 8*gb, 8*gb, 1)}}
	ok, no := c.Installable(nil, DefaultMargins())
	if len(ok) != 0 || !strings.Contains(no[0].Reason, "no device") {
		t.Fatalf("got ok=%v no=%v", ok, no)
	}
}

func TestReasonsAreActionable(t *testing.T) {
	c := Catalog{Entries: []Entry{entry("big", 40*gb, 40*gb, 1)}}
	_, no := c.Installable([]Device{dev("3090", 24, false)}, DefaultMargins())
	r := no[0].Reason
	// It has to say the requirement AND what was found, or the reader cannot tell what to change.
	if !strings.Contains(r, "GiB") || !strings.Contains(r, "found") {
		t.Fatalf("a model silently missing from a list reads like a bug; the reason must name the requirement and the shortfall: %q", r)
	}
}

// --- validation ---

func TestParseRejectsUnknownProvenance(t *testing.T) {
	_, err := Parse([]byte(`{"entries":[{"id":"x","min_total_bytes":1,"provenance":"probably"}]}`))
	if err == nil {
		t.Fatal("an unrecognised provenance would default to something and decide installability silently")
	}
}

func TestParseRejectsAnEntryWithNoRequirement(t *testing.T) {
	_, err := Parse([]byte(`{"entries":[{"id":"x","provenance":"measured","streams":1,"total_context":1}]}`))
	if err == nil {
		t.Fatal("a zero requirement matches every machine")
	}
}

func TestParseRejectsAMeasuredEntryWithNoConfiguration(t *testing.T) {
	_, err := Parse([]byte(`{"entries":[{"id":"x","min_total_bytes":1,"provenance":"measured"}]}`))
	if err == nil {
		t.Fatal("a measurement without the settings that produced it cannot be reproduced")
	}
}

func TestParseRejectsDuplicateIDs(t *testing.T) {
	_, err := Parse([]byte(`{"entries":[
		{"id":"x","min_total_bytes":1,"provenance":"computed"},
		{"id":"x","min_total_bytes":1,"provenance":"computed"}]}`))
	if err == nil {
		t.Fatal("duplicate ids make an entry ambiguous")
	}
}

func TestParseRejectsMissingID(t *testing.T) {
	if _, err := Parse([]byte(`{"entries":[{"min_total_bytes":1,"provenance":"computed"}]}`)); err == nil {
		t.Fatal("want an error")
	}
}
