// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/sideblank/llama-herd/internal/catalog"
	"github.com/sideblank/llama-herd/internal/llama"
)

// modelsCmd reports which supported models this machine can install, and why the rest cannot.
//
// The "why" is the point. A model silently missing from a list tells the reader nothing and reads
// like a bug; naming the shortfall tells them what to change, or what to buy.
func modelsCmd(args []string) int {
	fs := newFlagSet("models")
	all := fs.Bool("all", false, "include models this machine cannot run, with the reason")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	cat, overridden, err := catalog.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "models:", err)
		return 1
	}
	if overridden {
		fmt.Fprintf(os.Stderr,
			"⚠️  using the catalog at %s — these combinations are not the supported set\n\n",
			os.Getenv(catalog.OverridePath))
	}

	// Backends register devices during init, so this has to come first or the list is empty.
	llama.Backend()
	defer llama.BackendFree()
	llama.LoadBackends()

	var devs []catalog.Device
	for _, d := range llama.GPUs() {
		devs = append(devs, catalog.Device{
			Name:       d.Name,
			TotalBytes: d.TotalBytes,
			// Apple Silicon shares its ceiling with the host, so it needs more headroom.
			// Detected from the backend name rather than from build tags, because a binary
			// built with Metal support may still run beside a discrete card.
			Unified: isUnified(d),
		})
	}

	fmt.Println("hardware")
	if len(devs) == 0 {
		fmt.Println("  none detected — no device with usable memory")
	}
	for _, d := range devs {
		kind := "dedicated"
		if d.Unified {
			kind = "unified (shared with the host)"
		}
		fmt.Printf("  %-28s %7.1f GiB  %s\n", d.Name, float64(d.TotalBytes)/(1<<30), kind)
	}
	fmt.Println()

	ok, no := cat.Installable(devs, catalog.DefaultMargins())

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "installable (%d)\n", len(ok))
	if len(ok) == 0 {
		fmt.Fprintln(w, "  none")
	}
	for _, f := range ok {
		fmt.Fprintf(w, "  %s\t%s\t%d streams x %d ctx\n",
			f.Entry.ID, f.Entry.Quant, f.Entry.Streams, f.Entry.TotalContext/f.Entry.Streams)
	}
	w.Flush()

	if *all && len(no) > 0 {
		fmt.Println()
		w = tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintf(w, "not installable (%d)\n", len(no))
		for _, f := range no {
			fmt.Fprintf(w, "  %s\t%s\n", f.Entry.ID, f.Reason)
		}
		w.Flush()
	} else if len(no) > 0 {
		fmt.Printf("\n%d not installable on this machine — `llama-herd models --all` for the reasons\n",
			len(no))
	}
	return 0
}

// isUnified reports whether a device's memory is shared with the host.
//
// Metal is the case that matters: it reports recommendedMaxWorkingSetSize, which is the right
// number, but that ceiling moves as the rest of the machine allocates.
func isUnified(d llama.Device) bool {
	for _, s := range []string{"Metal", "Apple"} {
		if containsFold(d.Name, s) || containsFold(d.Description, s) {
			return true
		}
	}
	return false
}

func containsFold(hay, needle string) bool {
	if len(needle) > len(hay) {
		return false
	}
	lower := func(b byte) byte {
		if b >= 'A' && b <= 'Z' {
			return b + 32
		}
		return b
	}
	for i := 0; i+len(needle) <= len(hay); i++ {
		match := true
		for j := 0; j < len(needle); j++ {
			if lower(hay[i+j]) != lower(needle[j]) {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
