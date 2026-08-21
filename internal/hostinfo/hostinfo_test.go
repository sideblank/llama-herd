package hostinfo

import "testing"

func TestReadDegradesRatherThanFailing(t *testing.T) {
	h := Read()
	if h.OS == "" || h.Arch == "" || h.CPUs < 1 {
		t.Fatalf("basic fields missing: %+v", h)
	}
	if h.Goroutines < 1 {
		t.Error("goroutine count should be at least 1")
	}
	if h.UptimeSeconds < 0 {
		t.Error("uptime should not be negative")
	}
	// Platform-specific fields may legitimately be zero; they must not panic or error.
	if h.LoadAvg1 < 0 || h.MemTotalBytes > 1<<62 {
		t.Errorf("implausible values: %+v", h)
	}
}

func TestOversubscribed(t *testing.T) {
	if !(Host{CPUs: 4, LoadAvg1: 9}).Oversubscribed() {
		t.Error("load of 9 on 4 cores is oversubscribed")
	}
	if (Host{CPUs: 16, LoadAvg1: 1}).Oversubscribed() {
		t.Error("load of 1 on 16 cores is not")
	}
	if (Host{CPUs: 8, LoadAvg1: 0}).Oversubscribed() {
		t.Error("an unavailable load average must not read as oversubscribed")
	}
}
