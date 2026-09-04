// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package vcontext

import (
	"strings"
	"testing"
)

func chunkOf(i int, text string) Chunk {
	return Chunk{Index: i, Text: text, Start: 0, End: len(text)}
}

// A shared rare term is a relationship; shared common words are not. That distinction is the whole
// value of the scorer, and getting it wrong spends spare streams bridging unrelated passages.
func TestRareSharedTermsOutrankCommonOnes(t *testing.T) {
	filler := "The system is deployed and the system is running and the system is available. "
	chunks := []Chunk{
		chunkOf(0, filler+"Acme Corporation signed the agreement."),
		chunkOf(1, filler+"Nothing of note occurred here at all."),
		chunkOf(2, filler+"The invoice was issued to Acme Corporation."),
		chunkOf(3, filler+"Routine maintenance completed successfully."),
	}
	related := RareTermRelated(chunks)

	shared := related(chunks[0], chunks[2]) // both name Acme
	common := related(chunks[1], chunks[3]) // share only filler
	if shared <= common {
		t.Errorf("chunks sharing a rare name scored %.2f against %.2f for chunks sharing only "+
			"boilerplate — inverse document frequency is not discriminating", shared, common)
	}
}

// Numbers must count. A port, an amount or a case number is exactly the shared detail that makes
// two distant passages worth bridging.
func TestNumbersAreSignificantTerms(t *testing.T) {
	chunks := []Chunk{
		chunkOf(0, "The service listens on port 8080 during normal operation."),
		chunkOf(1, "Unrelated discussion of scheduling and staffing arrangements."),
		chunkOf(2, "Traffic arriving on port 8080 is logged for audit purposes."),
	}
	related := RareTermRelated(chunks)
	if related(chunks[0], chunks[2]) <= related(chunks[0], chunks[1]) {
		t.Error("a shared port number did not outrank an unrelated chunk")
	}
}

// A chunk is not related to itself, or every pairing is dominated by self-matches.
func TestChunkIsNotRelatedToItself(t *testing.T) {
	chunks := []Chunk{chunkOf(0, "Acme Corporation"), chunkOf(1, "Globex Incorporated")}
	related := RareTermRelated(chunks)
	if got := related(chunks[0], chunks[0]); got != 0 {
		t.Errorf("self-relatedness = %v, want 0", got)
	}
}

// The score must be symmetric, or which chunk is named first changes what gets bridged.
func TestRelatednessIsSymmetric(t *testing.T) {
	chunks := []Chunk{
		chunkOf(0, "Acme Corporation and the termination clause."),
		chunkOf(1, "Filler about nothing in particular whatsoever."),
		chunkOf(2, "The termination clause binds Acme Corporation."),
	}
	related := RareTermRelated(chunks)
	if a, b := related(chunks[0], chunks[2]), related(chunks[2], chunks[0]); a != b {
		t.Errorf("asymmetric: %v vs %v", a, b)
	}
}

// End to end: the scorer must actually drive bridge selection toward the related pair.
func TestScorerDrivesLongRangeBridgeChoice(t *testing.T) {
	filler := strings.Repeat("Routine text of no particular significance. ", 4)
	chunks := []Chunk{
		chunkOf(0, "Acme Corporation is the counterparty. "+filler),
		chunkOf(1, filler),
		chunkOf(2, filler),
		chunkOf(3, "Payment is owed by Acme Corporation. "+filler),
	}
	pos := 0
	for i := range chunks {
		chunks[i].Start = pos
		pos += len(chunks[i].Text)
		chunks[i].End = pos
		chunks[i].Cut = CutParagraph
	}
	budget := StreamBudget{Total: 16, Chunks: 4, Spare: 6}
	bridges := PlanBridges(chunks, budget, 64, RareTermRelated(chunks))

	var found bool
	for _, b := range bridges {
		if !b.Adjacent && ((b.Left == 0 && b.Right == 3) || (b.Left == 3 && b.Right == 0)) {
			found = true
		}
	}
	if !found {
		t.Errorf("the two chunks naming Acme were not bridged; planned: %+v", bridges)
	}
}
