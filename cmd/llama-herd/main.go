// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

// Command llama-herd is the runtime's entry point.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/sideblank/llama-herd/internal/llama"
)

// Set at link time by the release build:
//
//	-ldflags "-X main.version=v1.2.3 -X main.commit=abc1234 -X main.date=..."
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
	// llamaCppRef is the upstream llama.cpp tag this binary was built against. A
	// llama-herd build is really a pair — this code plus that upstream — because the
	// library API and the GGUF formats it reads both move. Reporting it turns "the
	// model will not load" into an answerable question.
	llamaCppRef = "unknown"
)

// osName and archName keep the runtime import in one place.
func osName() string   { return runtime.GOOS }
func archName() string { return runtime.GOARCH }

func usage() {
	fmt.Fprintf(os.Stderr, `llama-herd %s

usage:
  llama-herd serve --manifest <file> [--addr <addr>]
                         host the manifest's models over an OpenAI-compatible API
  llama-herd bench --manifest <file> [--model <name>] [--streams 1,2,4,8]
                         measure throughput and emit a reproducible report
  llama-herd sweep --manifest <file> [--streams 4,6,8] [--kv q8_0/q8_0,q8_0/q4_0]
                         measure a matrix of configurations against one resident copy of
                         the weights, so a rented card pays the model load once
  llama-herd models [--all]
                         list the models this machine can install, and why the rest cannot
  llama-herd inspect [--all] <model.gguf>
                         report what a model file declares, including MTP layers
  llama-herd fit [--card 3090] [--streams 4] [--context 128k] <model.gguf>
                         report what streams x context actually fit on a card
  llama-herd tasks --request <text> [--dry-run] [--streams 8]
                         decompose a request into a task graph, run it in dependency order,
                         and assemble the answer; --dry-run prints the plan and stops
  llama-herd vcontext --doc <file> [--query <q>] [--streams 8]
                         process an input of any size across streams and merge the results
  llama-herd canon --model <gguf> --doc <file> [--json]
                         measure what each canonicalisation pass costs in real tokens
  llama-herd latent-probe --model <small.gguf> [--control] [--as-tokens] [--scale 1.0]
                         test whether a chunk's final hidden state can seed a grounded
                         generation, the premise HLSR rests on
  llama-herd standby [--addr :8080] [--status <file>]
                         hold the port and report boot progress, so a long preparation is
                         not mistaken for an unhealthy instance
  llama-herd version     print version and build information
  llama-herd doctor      verify linkage and list the devices it can see

`, version)
}

// newFlagSet builds a subcommand flag set that prints usage to stderr on error.
func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	return fs
}

func main() {
	flag.Usage = usage
	flag.Parse()

	switch flag.Arg(0) {
	case "serve":
		os.Exit(serve(flag.Args()[1:]))

	case "bench":
		os.Exit(benchCmd(flag.Args()[1:]))

	case "sweep":
		os.Exit(sweep(flag.Args()[1:]))

	case "standby":
		os.Exit(standby(flag.Args()[1:]))

	case "latent-probe":
		os.Exit(latentProbe(flag.Args()[1:]))

	case "vcontext":
		os.Exit(vcontextCmd(flag.Args()[1:]))

	case "tasks":
		os.Exit(tasksCmd(flag.Args()[1:]))

	case "canon":
		os.Exit(canonCmd(flag.Args()[1:]))

	case "models":
		os.Exit(modelsCmd(flag.Args()[1:]))

	case "inspect":
		os.Exit(inspect(flag.Args()[1:]))

	case "fit":
		os.Exit(fitCmd(flag.Args()[1:]))

	case "version":
		fmt.Printf("llama-herd %s\n", version)
		fmt.Printf("  commit:   %s\n", commit)
		fmt.Printf("  built:    %s\n", date)
		fmt.Printf("  go:       %s\n", runtime.Version())
		fmt.Printf("  platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
		fmt.Printf("  llama.cpp: %s\n", llamaCppRef)

	case "doctor":
		// Proves the binary actually links and initialises libllama, which is the
		// failure a release archive is most likely to ship broken: a binary that
		// builds fine but cannot find its library at run time.
		llama.Backend()
		defer llama.BackendFree()
		fmt.Printf("llama.cpp backend initialised — linkage OK (%s/%s)\n", runtime.GOOS, runtime.GOARCH)
		fmt.Printf("built against: %s\n", llamaCppRef)
		fmt.Printf("system: %s\n", llama.SystemInfo())

		devs := llama.Devices()
		fmt.Printf("\ndevices (%d, max split across %d):\n", len(devs), llama.MaxDevices())
		for _, d := range devs {
			fmt.Printf("  [%d] %-8s %-12s %s\n", d.Index, d.Type, d.Name, d.Description)
			if d.TotalBytes > 0 {
				fmt.Printf("       memory: %.1f GiB free of %.1f GiB\n",
					float64(d.FreeBytes)/(1<<30), float64(d.TotalBytes)/(1<<30))
			}
		}
		if len(llama.GPUs()) == 0 {
			fmt.Println("\n  no dedicated-memory GPU found — this build will run on CPU")
		}

	default:
		usage()
		os.Exit(2)
	}
}
