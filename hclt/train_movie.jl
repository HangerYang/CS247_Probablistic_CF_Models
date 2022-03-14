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

function dataset_cpu(;dataset_path="../data",
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

function dataset_gpu(dataset_name)
    cu.(dataset_cpu(;dataset_name))
end

function run(; batch_size = 512, num_epochs1 = 100, num_epochs2 = 100, num_epochs3 = 20,
             pseudocount = 0.1, latents = 32, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.95)
    train, valid, test = dataset_cpu()
    train_gpu, valid_gpu, test_gpu = dataset_gpu("movielens1M_missing")

    q = 100
    no_missing_train = Array{Union{Missing, Int32}}(missing, q, size(train)[2])
    for i in 1:q
        for j in 1:size(train)[2]
            no_missing_train[i, j] = train[i, j]
            if ismissing(train[i, j])
                no_missing_train[i, j] = 0
            end
        end
    end

    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(no_missing_train, latents; num_cats = 2, pseudocount = 0.01, input_type = Literal);
    init_parameters(pc; perturbation = 0.4);
    println("Number of free parameters: $(num_parameters(pc))")

    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)

    softness    = 0
    @time mini_batch_em(bpc, train_gpu, num_epochs1; batch_size, pseudocount,
    			 softness, param_inertia = param_inertia1, param_inertia_end = param_inertia2, debug = false)

    ll1 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll1)")

    @time mini_batch_em(bpc, train_gpu, num_epochs2; batch_size, pseudocount,
    			 softness, param_inertia = param_inertia2, param_inertia_end = param_inertia3)

    ll2 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll2)")

    @time full_batch_em(bpc, train_gpu, num_epochs3; batch_size, pseudocount, softness)

    ll3 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll3)")

    print("update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc)

    write("./movielens1M.jpc", pc)

    ll1, ll2, ll3, batch_size, pseudocount, latents
end

run()