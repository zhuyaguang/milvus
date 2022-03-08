#pragma once

#include <memory>

#include "NANG/include/efanna2e/index_graph.h"
#include <knowhere/common/Exception.h>
#include <knowhere/index/IndexType.h>
#include <knowhere/index/vector_index/VecIndex.h>

namespace milvus {
namespace knowhere {

class IndexNANG : public VecIndex {
 public:
 IndexNANG()
 {
     std::cout<<"construct IndexNANG"<<std::endl;
 }
    BinarySet
    Serialize(const Config& config) override;

    void
    Load(const BinarySet& index_array) override;

    void
    BuildAll(const DatasetPtr&, const Config&) override;

    void
    Train(const DatasetPtr& dataset_ptr, const Config& config) override {
        KNOWHERE_THROW_MSG("NANG not support build item dynamically, please invoke BuildAll interface.");
    }

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override {
        KNOWHERE_THROW_MSG("Incremental index NANG is not supported");
    }
    DatasetPtr
    Query(const DatasetPtr& dataset_ptr, const Config& config,const faiss::BitsetView bitset) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    void
    UpdateIndexSize() override;

 private:
    std::shared_ptr<efanna2e::IndexGraph> index_;
};

}  // namespace knowhere
}  // namespace milvus
