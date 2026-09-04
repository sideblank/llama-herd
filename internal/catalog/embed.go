// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package catalog

import (
	_ "embed"
	"fmt"
	"os"
)

//go:embed models.json
var builtin []byte

// Builtin returns the catalog compiled into the binary.
func Builtin() (Catalog, error) { return Parse(builtin) }

// OverridePath is the environment variable naming a replacement catalog.
//
// An escape hatch on purpose: the library is closed, but someone with hardware we have not measured
// should be able to try a combination at their own risk rather than be unable to run at all. It is
// opt-in, explicit, and names a file — nothing is loaded implicitly.
const OverridePath = "LLAMA_HERD_CATALOG"

// Load returns the override catalog when OverridePath is set, otherwise the builtin.
//
// An unreadable or malformed override is an ERROR, never a silent fall back to the builtin. Falling
// back would run a configuration the operator did not choose while reporting success, which is the
// failure this whole package exists to prevent.
func Load() (Catalog, bool, error) {
	path := os.Getenv(OverridePath)
	if path == "" {
		c, err := Builtin()
		return c, false, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Catalog{}, true, fmt.Errorf("catalog: %s=%s could not be read: %w",
			OverridePath, path, err)
	}
	c, err := Parse(data)
	if err != nil {
		return Catalog{}, true, fmt.Errorf("catalog: %s=%s: %w", OverridePath, path, err)
	}
	return c, true, nil
}
