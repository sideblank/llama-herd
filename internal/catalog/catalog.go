// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Package catalog holds the supported (model x hardware) combinations and decides what a given
// machine can install.
//
// The library is closed by design. Every throughput and capacity figure this project publishes is
// specific to a model, a quantisation, a build and a card; a runtime that accepts arbitrary
// combinations can make none of those promises, because it has measured none of them.
package catalog

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Provenance says how an entry's requirements were established.
//
// ⛔ Load-bearing. An install gate must consult Measured entries only. 128 streams FIT the arithmetic
// on a 3090 and took the container with it; 72 fit and collapsed on one node while running on
// another. A gate keyed to a fit calculation would have accepted both. The error runs the other way
// too: a conservative computed gate refuses configurations that work, and for a product that is
// worse than a crash, because the user never learns it would have been fine.
type Provenance string

const (
	// Measured means this configuration was run on this hardware and the numbers came off it.
	Measured Provenance = "measured"
	// Computed means it fits a capacity calculation and nothing has run it. A candidate, not a
	// supported configuration.
	Computed Provenance = "computed"
)

// Entry is one supported combination.
type Entry struct {
	ID    string `json:"id"`
	Model string `json:"model"`
	Quant string `json:"quant"`

	// MinTotalBytes is the device memory required across all devices.
	MinTotalBytes uint64 `json:"min_total_bytes"`
	// MinPerDeviceBytes is what EACH participating device must have.
	//
	// Independent of the total and not derivable from it: a model needing 40 GB across two
	// devices does not necessarily run on six 12 GB cards, because a layer must fit on the
	// device holding it.
	MinPerDeviceBytes uint64 `json:"min_per_device_bytes"`
	// MinDevices is how many devices the weights must be spread across.
	MinDevices int `json:"min_devices"`
	// SplitMode is the placement this entry was established with, where it matters.
	//
	// Recorded because three cards are not one large card: row-splitting depends on the
	// interconnect and can lose to layer-splitting on PCIe without NVLink, so the same three
	// devices can be a supported row for one mode and not the other.
	SplitMode string `json:"split_mode,omitempty"`

	// Streams and TotalContext are the configuration to run, not the maximum that fits.
	Streams      uint32 `json:"streams"`
	TotalContext uint32 `json:"total_context"`
	KVTypeK      string `json:"kv_type_k"`
	KVTypeV      string `json:"kv_type_v"`

	Provenance Provenance `json:"provenance"`
	// Note carries what a reader needs that the numbers do not say.
	Note string `json:"note,omitempty"`
}

// Catalog is the whole library.
type Catalog struct {
	Entries []Entry `json:"entries"`
}

// Device is one piece of hardware the catalog is matched against.
//
// Mirrors llama.Device without importing it, so catalog logic carries no cgo and can be tested
// without a backend.
type Device struct {
	Name string
	// TotalBytes is what the backend reports as usable.
	//
	// ⚠️ Means slightly different things per platform and both are correct: on a discrete card it
	// is the card's memory; on Apple Silicon it is `recommendedMaxWorkingSetSize`, roughly 70-75%
	// of unified memory, which is the safe allocation ceiling rather than the RAM figure.
	TotalBytes uint64
	// Unified marks memory shared with the host, which needs a larger safety margin: the ceiling
	// is shared with everything else running, so a browser taking 8 GB mid-run is the difference
	// between an install that works and one that fails on a bad afternoon.
	Unified bool
}

// Fit is the outcome of matching one entry against a machine.
type Fit struct {
	Entry Entry
	OK    bool
	// Reason says why an entry does not fit, in terms the reader can act on.
	//
	// A model silently missing from a list tells nobody anything and reads like a bug. "needs
	// 40 GB across >= 2 devices; found 24 GB on 1" tells someone what to change.
	Reason string
}

// Margins are the headroom fractions required beyond an entry's stated minimum.
type Margins struct {
	Dedicated float64
	Unified   float64
}

// DefaultMargins reserves more on unified memory than on a dedicated card, for the reason given on
// Device.Unified.
//
// ⚠️ Both numbers are judgement, not measurement. They are here as one named place to change rather
// than as established values.
func DefaultMargins() Margins { return Margins{Dedicated: 0.05, Unified: 0.20} }

// Installable partitions the catalog against the devices present.
//
// ⛔ Computed entries are never installable. They are candidates awaiting measurement; offering one
// is offering a configuration nobody has run.
func (c Catalog) Installable(devs []Device, m Margins) (ok []Fit, no []Fit) {
	gpus := make([]Device, 0, len(devs))
	for _, d := range devs {
		if d.TotalBytes > 0 {
			gpus = append(gpus, d)
		}
	}
	// Largest first: a model needing two devices should be matched against the two biggest.
	sort.SliceStable(gpus, func(i, j int) bool { return gpus[i].TotalBytes > gpus[j].TotalBytes })

	entries := append([]Entry(nil), c.Entries...)
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].ID < entries[j].ID })

	for _, e := range entries {
		f := e.fit(gpus, m)
		if f.OK {
			ok = append(ok, f)
		} else {
			no = append(no, f)
		}
	}
	return ok, no
}

func (e Entry) fit(gpus []Device, m Margins) Fit {
	if e.Provenance != Measured {
		return Fit{Entry: e, Reason: "not a supported configuration — fits a capacity " +
			"calculation but has never been run on this hardware"}
	}
	if len(gpus) == 0 {
		return Fit{Entry: e, Reason: "no device with usable memory was found"}
	}

	need := e.MinDevices
	if need < 1 {
		need = 1
	}
	if len(gpus) < need {
		return Fit{Entry: e, Reason: fmt.Sprintf("needs %d devices, found %d", need, len(gpus))}
	}

	// Each participating device must clear the per-device floor with margin. Checked before the
	// total, because it is the constraint people are surprised by: six small cards can beat the
	// total and still not hold a layer.
	usable := func(d Device) uint64 {
		f := m.Dedicated
		if d.Unified {
			f = m.Unified
		}
		return uint64(float64(d.TotalBytes) * (1 - f))
	}

	participating := gpus[:need]
	var total uint64
	for _, d := range participating {
		u := usable(d)
		if e.MinPerDeviceBytes > 0 && u < e.MinPerDeviceBytes {
			scope := "the device needs"
			if need > 1 {
				scope = fmt.Sprintf("each of %d devices needs", need)
			}
			return Fit{Entry: e, Reason: fmt.Sprintf(
				"%s %s usable; found %s on %q after margin",
				scope, gib(e.MinPerDeviceBytes), gib(u), d.Name)}
		}
		total += u
	}
	if total < e.MinTotalBytes {
		return Fit{Entry: e, Reason: fmt.Sprintf(
			"needs %s across %s%s; found %s usable on %s",
			gib(e.MinTotalBytes), atLeast(need, len(gpus)), plural(need), gib(total), plural(len(gpus)))}
	}
	return Fit{Entry: e, OK: true}
}

func atLeast(need, have int) string {
	if have > need {
		return ">= "
	}
	return ""
}

func plural(n int) string {
	if n == 1 {
		return "1 device"
	}
	return fmt.Sprintf("%d devices", n)
}

func gib(b uint64) string {
	return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30))
}

// Parse reads a catalog and validates it.
//
// Validated on load rather than on use: a malformed entry that reaches the install gate either
// offers a configuration nobody measured or hides one that works, and neither failure announces
// itself.
func Parse(data []byte) (Catalog, error) {
	var c Catalog
	if err := json.Unmarshal(data, &c); err != nil {
		return Catalog{}, fmt.Errorf("catalog: %w", err)
	}
	seen := map[string]bool{}
	for i, e := range c.Entries {
		switch {
		case strings.TrimSpace(e.ID) == "":
			return Catalog{}, fmt.Errorf("catalog: entry %d has no id", i)
		case seen[e.ID]:
			return Catalog{}, fmt.Errorf("catalog: duplicate id %q", e.ID)
		case e.Provenance != Measured && e.Provenance != Computed:
			return Catalog{}, fmt.Errorf("catalog: entry %q has provenance %q, want %q or %q",
				e.ID, e.Provenance, Measured, Computed)
		case e.MinTotalBytes == 0:
			return Catalog{}, fmt.Errorf("catalog: entry %q states no memory requirement, so it "+
				"would match every machine", e.ID)
		case e.Provenance == Measured && (e.Streams == 0 || e.TotalContext == 0):
			return Catalog{}, fmt.Errorf("catalog: measured entry %q carries no configuration to "+
				"run — a measurement without the settings that produced it cannot be reproduced",
				e.ID)
		}
		seen[e.ID] = true
	}
	return c, nil
}
