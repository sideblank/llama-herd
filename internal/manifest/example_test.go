package manifest

import "testing"

func TestExampleManifestIsValid(t *testing.T) {
	m, err := Load("../../examples/manifest.json")
	if err != nil {
		t.Fatalf("the shipped example must validate: %v", err)
	}
	if len(m.Models) != 2 {
		t.Fatalf("models = %d, want 2", len(m.Models))
	}
	if m.Models[1].SplitMode != SplitLayer || len(m.Models[1].TensorSplit) != 2 {
		t.Fatalf("second model should demonstrate a split: %+v", m.Models[1])
	}
}
