package manifest

import (
	"os"
	"path/filepath"
	"testing"
)

// The shipped example is what people copy, so it has to satisfy the rules this package
// enforces. An example that the loader refuses teaches a configuration that cannot start; one
// that merely parses can still teach a slow one, which is why the unified-pool rules exist.
func TestShippedExampleIsValid(t *testing.T) {
	for _, name := range []string{"manifest.json", "3090-throughput.json"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join("..", "..", "examples", name)
			f, err := os.Open(path)
			if err != nil {
				t.Skipf("no example to check: %v", err)
			}
			defer f.Close()
			m, err := Parse(f)
			if err != nil {
				t.Fatalf("the shipped example does not load: %v", err)
			}
			for _, mm := range m.Models {
				if mm.Streams > 1 && !mm.KVUnified {
					t.Errorf("%s: the example runs %d streams with a pool per stream, which "+
						"makes the library decode once per sequence — it would teach the slow "+
						"arrangement", mm.Name, mm.Streams)
				}
				// A request admitted at the full per-stream share has nowhere to put its
				// answer. The example that documents a measured profile is the one most
				// likely to be copied verbatim, so the headroom has to be real.
				if mm.Streams > 0 && mm.AdmitContext > 0 {
					if share := mm.Context / mm.Streams; mm.AdmitContext >= share {
						t.Errorf("%s: admits %d tokens into a %d-token per-stream share, "+
							"leaving nothing to generate into",
							mm.Name, mm.AdmitContext, share)
					}
				}
			}
		})
	}
}
