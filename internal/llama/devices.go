// Copyright 2026 the llama-herd authors
// SPDX-License-Identifier: Apache-2.0

package llama

/*
#include <stdlib.h>
#include "llama.h"
#include "ggml-backend.h"
*/
import "C"

import "unsafe"

// DeviceType distinguishes what a backend device actually is. A heterogeneous host can
// present several at once, and they are not interchangeable: an integrated GPU shares host
// memory, so its reported capacity says nothing about how much a model can safely take.
type DeviceType int32

const (
	DeviceCPU   DeviceType = C.GGML_BACKEND_DEVICE_TYPE_CPU
	DeviceGPU   DeviceType = C.GGML_BACKEND_DEVICE_TYPE_GPU
	DeviceIGPU  DeviceType = C.GGML_BACKEND_DEVICE_TYPE_IGPU
	DeviceAccel DeviceType = C.GGML_BACKEND_DEVICE_TYPE_ACCEL
	DeviceMeta  DeviceType = C.GGML_BACKEND_DEVICE_TYPE_META
)

func (t DeviceType) String() string {
	switch t {
	case DeviceCPU:
		return "cpu"
	case DeviceGPU:
		return "gpu"
	case DeviceIGPU:
		return "igpu"
	case DeviceAccel:
		return "accel"
	case DeviceMeta:
		return "meta"
	default:
		return "unknown"
	}
}

// Device describes one backend device. FreeBytes is a snapshot: it moves as other processes
// allocate, so treat it as a budget input rather than a guarantee.
type Device struct {
	Index       int
	Name        string
	Description string
	Type        DeviceType
	FreeBytes   uint64
	TotalBytes  uint64
}

// IsGPU reports whether the device has dedicated memory worth placing weights in. Integrated
// GPUs are excluded deliberately: their memory is the host's.
func (d Device) IsGPU() bool { return d.Type == DeviceGPU }

// Devices enumerates the backend devices visible to the linked libllama.
//
// Call Backend() first — devices are registered during backend init, so an earlier call
// returns an empty list rather than an error.
//
// This is what makes a heterogeneous host tractable: cards of different capacities can be
// sized individually instead of assuming one uniform budget.
func Devices() []Device {
	n := int(C.ggml_backend_dev_count())
	out := make([]Device, 0, n)
	for i := 0; i < n; i++ {
		d := C.ggml_backend_dev_get(C.size_t(i))
		if d == nil {
			continue
		}
		var free, total C.size_t
		C.ggml_backend_dev_memory(d, &free, &total)
		out = append(out, Device{
			Index:       i,
			Name:        C.GoString(C.ggml_backend_dev_name(d)),
			Description: C.GoString(C.ggml_backend_dev_description(d)),
			Type:        DeviceType(C.ggml_backend_dev_type(d)),
			FreeBytes:   uint64(free),
			TotalBytes:  uint64(total),
		})
	}
	return out
}

// GPUs returns only devices with dedicated memory, in enumeration order.
func GPUs() []Device {
	all := Devices()
	out := make([]Device, 0, len(all))
	for _, d := range all {
		if d.IsGPU() {
			out = append(out, d)
		}
	}
	return out
}

// MaxDevices is the largest number of devices libllama will split across, and therefore the
// required length of a TensorSplit slice.
func MaxDevices() int { return int(C.llama_max_devices()) }

// cTensorSplit copies proportions into C memory sized to MaxDevices. The returned free
// function must be called once the load has finished.
//
// The array is C-allocated rather than a Go slice pointer because libllama holds it for the
// duration of the load, which is exactly the case cgo's pointer rules do not permit.
func cTensorSplit(split []float32) (*C.float, func()) {
	if len(split) == 0 {
		return nil, func() {}
	}
	n := MaxDevices()
	if len(split) > n {
		split = split[:n]
	}
	buf := C.calloc(C.size_t(n), C.sizeof_float)
	arr := unsafe.Slice((*float32)(buf), n)
	copy(arr, split)
	return (*C.float)(buf), func() { C.free(buf) }
}
