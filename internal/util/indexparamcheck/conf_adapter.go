// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package indexparamcheck

import (
    "fmt"
	"strconv"

	"github.com/milvus-io/milvus/internal/util/funcutil"
)

const (
	// L2 represents Euclidean distance
	L2 = "L2"

	// IP represents inner product distance
	IP = "IP"

	// HAMMING represents hamming distance
	HAMMING = "HAMMING"

	// JACCARD represents jaccard distance
	JACCARD = "JACCARD"

	// TANIMOTO represents tanimoto distance
	TANIMOTO = "TANIMOTO"

	// SUBSTRUCTURE represents substructure distance
	SUBSTRUCTURE = "SUBSTRUCTURE"

	// SUPERSTRUCTURE represents superstructure distance
	SUPERSTRUCTURE = "SUPERSTRUCTURE"

	MinNBits     = 1
	MaxNBits     = 16
	DefaultNBits = 8

	// MinNList is the lower limit of nlist that used in Index IVFxxx
	MinNList = 1
	// MaxNList is the upper limit of nlist that used in Index IVFxxx
	MaxNList = 65536

	// DefaultMinDim is the smallest dimension supported in Milvus
	DefaultMinDim = 1
	// DefaultMaxDim is the largest dimension supported in Milvus
	DefaultMaxDim = 32768

	NgtMinEdgeSize = 1
	NgtMaxEdgeSize = 200

	HNSWMinEfConstruction = 8
	HNSWMaxEfConstruction = 512
	HNSWMinM              = 4
	HNSWMaxM              = 64

	MinKNNG              = 5
	MaxKNNG              = 300
	MinSearchLength      = 10
	MaxSearchLength      = 300
	MinOutDegree         = 5
	MaxOutDegree         = 300
	MinCandidatePoolSize = 50
	MaxCandidatePoolSize = 1000

	MinNTrees = 1
	// too large of n_trees takes much time, if there is real requirement, change this threshold.
	MaxNTrees = 1024

	// DIM is a constant used to represent dimension
	DIM = "dim"
	// Metric is a constant used to metric type
	Metric = "metric_type"
	// NLIST is a constant used to nlist in Index IVFxxx
	NLIST = "nlist"
	NBITS = "nbits"
	IVFM  = "m"

	KNNG         = "knng"
	SearchLength = "search_length"
	OutDegree    = "out_degree"
	CANDIDATE    = "candidate_pool_size"

	EFConstruction = "efConstruction"
	HNSWM          = "M"

	K = "K"
	iter =" iter"

	PQM    = "PQM"
	NTREES = "n_trees"

	EdgeSize                  = "edge_size"
	ForcedlyPrunedEdgeSize    = "forcedly_pruned_edge_size"
	SelectivelyPrunedEdgeSize = "selectively_pruned_edge_size"

	OutgoingEdgeSize = "outgoing_edge_size"
	IncomingEdgeSize = "incoming_edge_size"

	IndexMode = "index_mode"
	CPUMode   = "CPU"
	GPUMode   = "GPU"
)

// METRICS is a set of all metrics types supported for float vector.
var METRICS = []string{L2, IP} // const

// BinIDMapMetrics is a set of all metric types supported for binary vector.
var BinIDMapMetrics = []string{HAMMING, JACCARD, TANIMOTO, SUBSTRUCTURE, SUPERSTRUCTURE}   // const
var BinIvfMetrics = []string{HAMMING, JACCARD, TANIMOTO}                                   // const
var supportDimPerSubQuantizer = []int{32, 28, 24, 20, 16, 12, 10, 8, 6, 4, 3, 2, 1}        // const
var supportSubQuantizer = []int{96, 64, 56, 48, 40, 32, 28, 24, 20, 16, 12, 8, 4, 3, 2, 1} // const

type ConfAdapter interface {
	// CheckTrain returns true if the index can be built with the specific index parameters.
	CheckTrain(map[string]string) bool
}

// BaseConfAdapter checks if a `FLAT` index can be built.
type BaseConfAdapter struct {
}

// CheckTrain check whether the params contains supported metrics types
func (adapter *BaseConfAdapter) CheckTrain(params map[string]string) bool {
	// dimension is specified when create collection
	//if !CheckIntByRange(params, DIM, DefaultMinDim, DefaultMaxDim) {
	//	return false
	//}

	return CheckStrByValues(params, Metric, METRICS)
}

func newBaseConfAdapter() *BaseConfAdapter {
	return &BaseConfAdapter{}
}

// IVFConfAdapter checks if a IVF index can be built.
type IVFConfAdapter struct {
	BaseConfAdapter
}

// CheckTrain returns true if the index can be built with the specific index parameters.
func (adapter *IVFConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, NLIST, MinNList, MaxNList) {
		return false
	}

	// skip check number of rows

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newIVFConfAdapter() *IVFConfAdapter {
	return &IVFConfAdapter{}
}

// IVFPQConfAdapter checks if a IVF_PQ index can be built.
type IVFPQConfAdapter struct {
	IVFConfAdapter
}

// CheckTrain checks if ivf-pq index can be built with the specific index parameters.
func (adapter *IVFPQConfAdapter) CheckTrain(params map[string]string) bool {
	if !adapter.IVFConfAdapter.CheckTrain(params) {
		return false
	}

	return adapter.checkPQParams(params)
}

func (adapter *IVFPQConfAdapter) checkPQParams(params map[string]string) bool {
	dimStr, dimensionExist := params[DIM]
	if !dimensionExist { // dimension is specified when creating collection
		return true
	}

	dimension, err := strconv.Atoi(dimStr)
	if err != nil { // invalid dimension
		return false
	}

	// nbits can be set to default: 8
	nbitsStr, nbitsExist := params[NBITS]
	var nbits int
	if !nbitsExist {
		nbits = 8
	} else {
		nbits, err = strconv.Atoi(nbitsStr)
		if err != nil { // invalid nbits
			return false
		}
	}

	mStr, ok := params[IVFM]
	if !ok {
		return false
	}
	m, err := strconv.Atoi(mStr)
	if err != nil { // invalid m
		return false
	}

	mode, ok := params[IndexMode]
	if !ok {
		mode = CPUMode
	}

	if mode == GPUMode && !adapter.checkGPUPQParams(dimension, m, nbits) {
		return false
	}

	return adapter.checkCPUPQParams(dimension, m)
}

func (adapter *IVFPQConfAdapter) checkGPUPQParams(dimension, m, nbits int) bool {
	/*
	 * Faiss 1.6
	 * Only 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32 dims per sub-quantizer are currently supported with
	 * no precomputed codes. Precomputed codes supports any number of dimensions, but will involve memory overheads.
	 */

	subDim := dimension / m
	return funcutil.SliceContain(supportSubQuantizer, m) && funcutil.SliceContain(supportDimPerSubQuantizer, subDim) && nbits == 8
}

func (adapter *IVFPQConfAdapter) checkCPUPQParams(dimension, m int) bool {
	return (dimension % m) == 0
}

func newIVFPQConfAdapter() *IVFPQConfAdapter {
	return &IVFPQConfAdapter{}
}

// IVFSQConfAdapter checks if a IVF_SQ index can be built.
type IVFSQConfAdapter struct {
	IVFConfAdapter
}

// CheckTrain returns true if the index can be built with the specific index parameters.
func (adapter *IVFSQConfAdapter) CheckTrain(params map[string]string) bool {
	params[NBITS] = strconv.Itoa(DefaultNBits)
	return adapter.IVFConfAdapter.CheckTrain(params)
}

func newIVFSQConfAdapter() *IVFSQConfAdapter {
	return &IVFSQConfAdapter{}
}

type BinIDMAPConfAdapter struct {
}

// CheckTrain checks if a binary flat index can be built with the specific parameters.
func (adapter *BinIDMAPConfAdapter) CheckTrain(params map[string]string) bool {
	// dimension is specified when create collection
	//if !CheckIntByRange(params, DIM, DefaultMinDim, DefaultMaxDim) {
	//	return false
	//}

	return CheckStrByValues(params, Metric, BinIDMapMetrics)
}

func newBinIDMAPConfAdapter() *BinIDMAPConfAdapter {
	return &BinIDMAPConfAdapter{}
}

// BinIVFConfAdapter checks if a bin IFV index can be built.
type BinIVFConfAdapter struct {
}

// CheckTrain checks if a binary ivf index can be built with specific parameters.
func (adapter *BinIVFConfAdapter) CheckTrain(params map[string]string) bool {
	// dimension is specified when create collection
	//if !CheckIntByRange(params, DIM, DefaultMinDim, DefaultMaxDim) {
	//	return false
	//}

	if !CheckIntByRange(params, NLIST, MinNList, MaxNList) {
		return false
	}

	if !CheckStrByValues(params, Metric, BinIvfMetrics) {
		return false
	}

	// skip checking the number of rows

	return true
}

func newBinIVFConfAdapter() *BinIVFConfAdapter {
	return &BinIVFConfAdapter{}
}

type NSGConfAdapter struct {
}

// CheckTrain checks if a nsg index can be built with specific parameters.
func (adapter *NSGConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckStrByValues(params, Metric, METRICS) {
		return false
	}

	if !CheckIntByRange(params, KNNG, MinKNNG, MaxKNNG) {
		return false
	}

	if !CheckIntByRange(params, SearchLength, MinSearchLength, MaxSearchLength) {
		return false
	}

	if !CheckIntByRange(params, OutDegree, MinOutDegree, MaxOutDegree) {
		return false
	}

	if !CheckIntByRange(params, CANDIDATE, MinCandidatePoolSize, MaxCandidatePoolSize) {
		return false
	}

	// skip checking the number of rows

	return true
}

func newNSGConfAdapter() *NSGConfAdapter {
	return &NSGConfAdapter{}
}

// HNSWConfAdapter checks if a hnsw index can be built.
type HNSWConfAdapter struct {
	BaseConfAdapter
}

// CheckTrain checks if a hnsw index can be built with specific parameters.
func (adapter *HNSWConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, EFConstruction, HNSWMinEfConstruction, HNSWMaxEfConstruction) {
		return false
	}

	if !CheckIntByRange(params, HNSWM, HNSWMinM, HNSWMaxM) {
		return false
	}

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newHNSWConfAdapter() *HNSWConfAdapter {
	return &HNSWConfAdapter{}
}

// HNSW2ConfAdapter checks if a hnsw index can be built.
type HNSW2ConfAdapter struct {
	BaseConfAdapter
}

// CheckTrain checks if a hnsw index can be built with specific parameters.
func (adapter *HNSW2ConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, EFConstruction, HNSWMinEfConstruction, HNSWMaxEfConstruction) {
		return false
	}

	if !CheckIntByRange(params, HNSWM, HNSWMinM, HNSWMaxM) {
		return false
	}

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newHNSW2ConfAdapter() *HNSW2ConfAdapter {
	return &HNSW2ConfAdapter{}
}

// NANGConfAdapter checks if a hnsw index can be built.
type NANGConfAdapter struct {
	BaseConfAdapter
}

// CheckTrain checks if a hnsw index can be built with specific parameters.
func (adapter *NANGConfAdapter) CheckTrain(params map[string]string) bool {
	// if !CheckIntByRange(params, K, 50, 500) {
    //    fmt.Println("CheckIntByRange1")
	// 	return false
	// }

	// if !CheckIntByRange(params, iter, 2, 30) {
    //    fmt.Println("CheckIntByRange2")
	// 	return false
	// }
    fmt.Println("CheckTrain")

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newNANGConfAdapter() *NANGConfAdapter {
	return &NANGConfAdapter{}
}

// ANNOYConfAdapter checks if an ANNOY index can be built.
type ANNOYConfAdapter struct {
	BaseConfAdapter
}

// CheckTrain checks if an annoy index can be built with specific parameters.
func (adapter *ANNOYConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, NTREES, MinNTrees, MaxNTrees) {
		return false
	}

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newANNOYConfAdapter() *ANNOYConfAdapter {
	return &ANNOYConfAdapter{}
}

// RHNSWFlatConfAdapter checks if a rhnsw flat index can be built.
type RHNSWFlatConfAdapter struct {
	BaseConfAdapter
}

// CheckTrain checks if a rhnsw flat index can be built with specific parameters.
func (adapter *RHNSWFlatConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, EFConstruction, HNSWMinEfConstruction, HNSWMaxEfConstruction) {
		return false
	}

	if !CheckIntByRange(params, HNSWM, HNSWMinM, HNSWMaxM) {
		return false
	}

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newRHNSWFlatConfAdapter() *RHNSWFlatConfAdapter {
	return &RHNSWFlatConfAdapter{}
}

// RHNSWPQConfAdapter checks if a rhnsw pq index can be built.
type RHNSWPQConfAdapter struct {
	BaseConfAdapter
	IVFPQConfAdapter
}

// CheckTrain checks if a rhnsw pq index can be built with specific parameters.
func (adapter *RHNSWPQConfAdapter) CheckTrain(params map[string]string) bool {
	if !adapter.BaseConfAdapter.CheckTrain(params) {
		return false
	}

	if !CheckIntByRange(params, EFConstruction, HNSWMinEfConstruction, HNSWMaxEfConstruction) {
		return false
	}

	if !CheckIntByRange(params, HNSWM, HNSWMinM, HNSWMaxM) {
		return false
	}

	dimension, _ := strconv.Atoi(params[DIM])
	pqmStr, ok := params[PQM]
	if !ok {
		return false
	}
	pqm, err := strconv.Atoi(pqmStr)
	if err != nil {
		return false
	}

	return adapter.IVFPQConfAdapter.checkCPUPQParams(dimension, pqm)
}

func newRHNSWPQConfAdapter() *RHNSWPQConfAdapter {
	return &RHNSWPQConfAdapter{}
}

// RHNSWSQConfAdapter checks if a rhnsw sq index can be built.
type RHNSWSQConfAdapter struct {
	BaseConfAdapter
}

// CheckTrain checks if a rhnsw sq index can be built with specific parameters.
func (adapter *RHNSWSQConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, EFConstruction, HNSWMinEfConstruction, HNSWMaxEfConstruction) {
		return false
	}

	if !CheckIntByRange(params, HNSWM, HNSWMinM, HNSWMaxM) {
		return false
	}

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newRHNSWSQConfAdapter() *RHNSWSQConfAdapter {
	return &RHNSWSQConfAdapter{}
}

// NGTPANNGConfAdapter checks if a NGT_PANNG index can be built.
type NGTPANNGConfAdapter struct {
	BaseConfAdapter
}

func (adapter *NGTPANNGConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, EdgeSize, NgtMinEdgeSize, NgtMaxEdgeSize) {
		return false
	}

	if !CheckIntByRange(params, ForcedlyPrunedEdgeSize, NgtMinEdgeSize, NgtMaxEdgeSize) {
		return false
	}

	if !CheckIntByRange(params, SelectivelyPrunedEdgeSize, NgtMinEdgeSize, NgtMaxEdgeSize) {
		return false
	}

	selectivelyPrunedEdgeSize, _ := strconv.Atoi(params[SelectivelyPrunedEdgeSize])
	forcedlyPrunedEdgeSize, _ := strconv.Atoi(params[ForcedlyPrunedEdgeSize])
	if selectivelyPrunedEdgeSize >= forcedlyPrunedEdgeSize {
		return false
	}

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newNGTPANNGConfAdapter() *NGTPANNGConfAdapter {
	return &NGTPANNGConfAdapter{}
}

// NGTONNGConfAdapter checks if a NGT_ONNG index can be built.
type NGTONNGConfAdapter struct {
	BaseConfAdapter
}

func (adapter *NGTONNGConfAdapter) CheckTrain(params map[string]string) bool {
	if !CheckIntByRange(params, EdgeSize, NgtMinEdgeSize, NgtMaxEdgeSize) {
		return false
	}

	if !CheckIntByRange(params, OutgoingEdgeSize, NgtMinEdgeSize, NgtMaxEdgeSize) {
		return false
	}

	if !CheckIntByRange(params, IncomingEdgeSize, NgtMinEdgeSize, NgtMaxEdgeSize) {
		return false
	}

	return adapter.BaseConfAdapter.CheckTrain(params)
}

func newNGTONNGConfAdapter() *NGTONNGConfAdapter {
	return &NGTONNGConfAdapter{}
}
