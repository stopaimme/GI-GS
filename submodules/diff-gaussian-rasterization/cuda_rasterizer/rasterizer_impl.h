/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		float3* pos_view;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	struct SampleState
	{
		size_t scan_size;
		uint32_t* bucket_to_tile;   // [total_buckets] maps bucket -> tile_id
		uint32_t* bucket_offsets;   // [num_tiles] inclusive prefix sum of per-tile bucket counts
		uint32_t* max_contrib;      // [num_tiles] per-tile max contributor index
		float* sampled_T;           // [total_buckets * BLOCK_SIZE] snapshot of T
		float* sampled_ar;          // [NUM_CHANNELS * total_buckets * BLOCK_SIZE] snapshot of accumulated color
		char* scanning_space;

		static SampleState fromChunk(char*& chunk, size_t total_buckets, size_t num_tiles);
	};

	template<typename T>
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	// Overload for SampleState which takes two size parameters
	template<>
	inline size_t required<SampleState>(size_t total_buckets)
	{
		// This is a placeholder; actual allocation uses fromChunk with both params.
		// We compute a conservative estimate.
		return 0;
	}

	// Helper to compute required size for SampleState
	inline size_t requiredSampleState(size_t total_buckets, size_t num_tiles)
	{
		char* size = nullptr;
		SampleState::fromChunk(size, total_buckets, num_tiles);
		return ((size_t)size) + 128;
	}
};