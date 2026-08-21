// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/json"
	"fmt"
)

// stopValue decodes the "stop" field, which clients send as either a single string or an
// array of them. Rejecting one of those shapes would break real callers, so both are taken.
type stopValue []string

func (s *stopValue) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		*s = nil
		return nil
	}
	switch b[0] {
	case '"':
		var one string
		if err := json.Unmarshal(b, &one); err != nil {
			return err
		}
		*s = []string{one}
		return nil
	case '[':
		var many []string
		if err := json.Unmarshal(b, &many); err != nil {
			return err
		}
		*s = many
		return nil
	default:
		return fmt.Errorf("stop must be a string or an array of strings")
	}
}
