using CUDA
using CSV
using DataFrames
using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods, full_batch_em, mini_batch_em, clt_edges2graphs, hclt_from_clt
using MLDatasets
using CUDA
using ChowLiuTrees: learn_chow_liu_tree
using Graphs: SimpleGraph, SimpleDiGraph, bfs_tree, center, neighbors,
connected_components, induced_subgraph, nv, add_edge!, rem_edge!
using MetaGraphs: get_prop, set_prop!, MetaDiGraph, vertices, indegree, outneighbors
using StatsBase: sample, Weights

device!(collect(devices())[4])

function dataset_cpu(;dataset_path="../data/",
    dataset_name="movielens1M_missing")

    function load(type)
        dataframe = CSV.read("$dataset_path/$dataset_name/$dataset_name.$type.data", DataFrame;
            header=false, types=Int32, strict=true)
                 # make sure the data is backed by a `BitArray`
        data_ = Tables.matrix(dataframe)
        data = Array{Union{Missing, Int32}}(missing, size(data_)[1], size(data_)[2])

        for i in 1:size(data_)[1]
            for j in 1:size(data_)[2]
                if data_[i, j] != -1
                    data[i, j] = data_[i, j]
                end
            end
        end
        data
    end
    train_cpu = load("train")
    valid_cpu = load("valid")
    test_cpu = load("test")

    train_cpu, valid_cpu, test_cpu
end

function dataset_gpu(dataset_name="movielens1M_missing")
    cu.(dataset_cpu(;dataset_name))
end

function eval(; batch_size = 512, num_epochs1 = 20, num_epochs2 = 20, num_epochs3 = 20,
             pseudocount = 0.1, latents = 32, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.95)
    train, valid, test = dataset_cpu()
    train_gpu, valid_gpu, test_gpu = dataset_gpu()

    pc = read("./movielens1M.jpc", ProbCircuit)


    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)

    neg_sample_num = 100
    k = 10

    n = size(test)[1]
    m = size(test)[2]
    acc = 0.0
    for i in 1:n
        test_example = test[i,:]
        test_example_batch = Array{Union{Int32, Missing}}(missing, 1+neg_sample_num, m)
        query_candidates = Vector{Int32}()
        neg_sample_candidates = Vector{Int32}()
        for j in 1:m
            if ismissing(test_example[j])
                push!(neg_sample_candidates, j)
            elseif test_example[j] == 1
                push!(query_candidates, j)
            end
        end

        query = sample(query_candidates)

        neg_samples = sample(neg_sample_candidates, neg_sample_num; replace=false)

        for i in 1:(1+neg_sample_num)
            test_example_batch[i,:] .= test_example
            if i > 1
                test_example_batch[i, query] = missing
                test_example_batch[i, neg_samples[i - 1]] = 1
            end
        end

        test_example_batch_gpu = cu(test_example_batch)
        lls = Array(loglikelihoods(bpc, test_example_batch_gpu; batch_size = 1024))
        if 1 in sortperm(lls, rev=true)[1:k]
            acc += 1.0
        end
    end

    println(acc /= n)
end

eval()