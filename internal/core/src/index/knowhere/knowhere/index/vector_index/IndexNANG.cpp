// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "knowhere/index/vector_index/IndexNANG.h"

#include <array>
#include <sstream>
#include <vector>

#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"


namespace milvus {
namespace knowhere {

BinarySet
IndexNANG::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    try {
        index_->saveIndex();
        std::shared_ptr<uint8_t[]> data((uint8_t*)(index_->getModelsave()));
        BinarySet res_set;
        res_set.Append("NANG", data, index_->getModelsize());
        if (config.contains(INDEX_FILE_SLICE_SIZE_IN_MEGABYTE)) {
            Disassemble(config[INDEX_FILE_SLICE_SIZE_IN_MEGABYTE].get<int64_t>() * 1024 * 1024, res_set);
        }
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexNANG::Load(const BinarySet& index_binary) {
    try {
        Assemble(const_cast<BinarySet&>(index_binary));
        auto binary = index_binary.GetByName("NANG");
        efanna2e::IndexRandom init_index(123,123);
        index_ = std::make_shared<efanna2e::IndexGraph>(efanna2e::L2, (efanna2e::Index *)(&init_index),123,123);
        index_->loadIndex((char*)(binary->data.get()));
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexNANG::BuildAll(const DatasetPtr& origin, const Config& config) {
    //std::cout<<"config : "<< config<<std::endl;
    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", config[knowhere::IndexParams::K].get<int64_t>());
    //std::cout<<"K : "<<paras.Get<unsigned>("K")<<std::endl;
    paras.Set<unsigned>("L", config[knowhere::IndexParams::L].get<int64_t>());
    //std::cout<<"L : "<<paras.Get<unsigned>("L")<<std::endl;
    paras.Set<unsigned>("iter", config[knowhere::IndexParams::iter].get<int64_t>());
    //std::cout<<"iter : "<<paras.Get<unsigned>("iter")<<std::endl;
    paras.Set<unsigned>("S",config[knowhere::IndexParams::S].get<int64_t>());
    //std::cout<<"S : "<<paras.Get<unsigned>("S")<<std::endl;
    paras.Set<unsigned>("R", config[knowhere::IndexParams::R].get<int64_t>());
    //std::cout<<"R : "<<paras.Get<unsigned>("R")<<std::endl;
    paras.Set<unsigned>("RANGE", config[knowhere::IndexParams::RANGE].get<int64_t>());
    //std::cout<<"RANGE : "<<paras.Get<unsigned>("RANGE")<<std::endl;
    paras.Set<unsigned>("PL",config[knowhere::IndexParams::PL].get<int64_t>());
    //std::cout<<"PL : "<<paras.Get<unsigned>("PL")<<std::endl;
    paras.Set<float>("B", config[knowhere::IndexParams::B].get<float>());
    //std::cout<<"B : "<<paras.Get<float>("B")<<std::endl;
    paras.Set<float>("M", config[knowhere::IndexParams::M_NANG].get<float>());
    //std::cout<<"M : "<<paras.Get<float>("M")<<std::endl;
    
    DatasetPtr dataset = origin;
    GET_TENSOR_DATA_DIM(dataset)
    efanna2e::IndexRandom init_index(dim,rows);
    index_ = std::make_shared<efanna2e::IndexGraph>(efanna2e::L2, (efanna2e::Index *)(&init_index),dim,rows);
    index_->Build(rows, (float*)p_data, paras);
}

DatasetPtr
IndexNANG::Query(const DatasetPtr& dataset_ptr, const Config& config,const faiss::BitsetView bitset) {
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", config[knowhere::IndexParams::search_L].get<int64_t>());
    GET_TENSOR_DATA_DIM(dataset_ptr);
    int search_k =  config[knowhere::meta::TOPK].get<int>();
    std::vector<std::vector<unsigned>> res(rows);
    std::vector<std::vector<float>> dis(rows);
    for (unsigned i = 0; i < rows; i++) {
        res[i].resize(search_k);
        dis[i].resize(search_k);
    }
#pragma omp parallel for
    for (auto i = 0; i < rows; ++i) {
        index_->SearchWithOptGraph(((float*)p_data) + i * dim, search_k, paras, res[i].data(), dis[i].data());
    }
    auto elems = search_k * rows;
    size_t p_id_size = sizeof(int64_t) * elems;
    size_t p_dist_size = sizeof(float) * elems;
    auto p_id = (int64_t*)malloc(p_id_size);
    auto p_dist = (float*)malloc(p_dist_size);
    auto ret_ds = std::make_shared<Dataset>();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < search_k; j++)
        {
            p_id[i * search_k + j] = res[i][j];
            p_dist[i * search_k + j] = dis[i][j];
        }
    }
    ret_ds->Set(meta::IDS, p_id);
    ret_ds->Set(meta::DISTANCE, p_dist);
    return ret_ds;
}

int64_t
IndexNANG::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->getNum();
}

int64_t
IndexNANG::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->getDim();
}

void
IndexNANG::UpdateIndexSize() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    index_size_ = index_->getModelsize();
}

}  // namespace knowhere
}  // namespace milvus
