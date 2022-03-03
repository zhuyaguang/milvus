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

#include <gtest/gtest.h>
#include "knowhere/common/Config.h"
#include "knowhere/index/vector_index/IndexNANG.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include <iostream>
#include <random>
#include "knowhere/common/Exception.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class NANGTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        IndexType = GetParam();
        std::cout << "IndexType from GetParam() is: " << IndexType << std::endl;
        Generate(64, 10000, 10);  // dim = 64, nb = 10000, nq = 10
        index_ = std::make_shared<milvus::knowhere::IndexNANG>();
        conf = milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, 64},          {milvus::knowhere::meta::TOPK, 10},
            {milvus::knowhere::IndexParams::K, 200},    {milvus::knowhere::IndexParams::L, 220},
            {milvus::knowhere::IndexParams::iter, 12},  {milvus::knowhere::IndexParams::S, 25},   
            {milvus::knowhere::IndexParams::R, 200},    {milvus::knowhere::IndexParams::RANGE, 40},
            {milvus::knowhere::IndexParams::PL, 50},    {milvus::knowhere::IndexParams::B, 1.0},   
            {milvus::knowhere::IndexParams::M, 1.0},    {milvus::knowhere::IndexParams::search_L, 20},
            {milvus::knowhere::Metric::TYPE, milvus::knowhere::Metric::L2},
        };
    }

 protected:
    milvus::knowhere::Config conf;
    std::shared_ptr<milvus::knowhere::IndexNANG> index_ = nullptr;
    std::string IndexType;
};

INSTANTIATE_TEST_CASE_P(NANGParameters, NANGTest, Values("NANG"));

TEST_P(NANGTest, NANG_basic) {
    assert(!xb.empty());
    index_->BuildAll(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    milvus::knowhere::BinarySet bs = index_->Serialize(conf);

    int64_t dim = base_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
    int64_t rows = base_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
    auto raw_data = base_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);
    milvus::knowhere::BinaryPtr bptr = std::make_shared<milvus::knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result, nq, k);

    // case: k > nb
    const int64_t new_rows = 6;
    base_dataset->Set(milvus::knowhere::meta::ROWS, new_rows);
    index_->BuildAll(base_dataset, conf);
    auto result2 = index_->Query(query_dataset, conf, nullptr);
    auto res_ids = result2->Get<int64_t*>(milvus::knowhere::meta::IDS);
    for (int64_t i = 0; i < nq; i++) {
        for (int64_t j = new_rows; j < k; j++) {
            ASSERT_EQ(res_ids[i * k + j], -1);
        }
    }
}

TEST_P(NANGTest, NANG_delete) {
    assert(!xb.empty());

    index_->BuildAll(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    faiss::ConcurrentBitsetPtr bitset = std::make_shared<faiss::ConcurrentBitset>(nb);
    for (auto i = 0; i < nq; ++i) {
        bitset->set(i);
    }

    // Serialize and Load before Query
    milvus::knowhere::BinarySet bs = index_->Serialize(conf);

    int64_t dim = base_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
    int64_t rows = base_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
    auto raw_data = base_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);
    milvus::knowhere::BinaryPtr bptr = std::make_shared<milvus::knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    auto result1 = index_->Query(query_dataset, conf, nullptr);
    AssertAnns(result1, nq, k);

    auto result2 = index_->Query(query_dataset, conf, bitset);
    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);
}


