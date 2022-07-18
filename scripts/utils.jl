

module Utils

export TimeSeriesSplit



struct TimeSeriesSplit
    n_splits::Int
    max_train_size::Int
    test_size::Int
    gap::Int
end

function TimeSeriesSplit(; n_splits=5, max_train_size=nothing, test_size=nothing, gap=0)
    @assert(n_splits >=2, "Number of splits must be at least 2")
    if isnothing(max_train_size)
        max_train_size = typemax(Int)
    end
    if isnothing(test_size)
        test_size = 0
    end
    return Splitter(n_splits, max_train_size, test_size, gap)
end

function (S::TimeSeriesSplit)(data::AbstractArray{T}) where{T}
    n_samples = size(data, 2)
    test_size = (S.test_size == 0) ? div(n_samples, S.n_splits+1) - S.gap : S.test_size
    test_splits = [n_samples - j*test_size + 1:n_samples - (j-1)*test_size for j = S.n_splits:-1:1]

    train_splits = [1 + split[1] - S.gap - min(i*div(n_samples, S.n_splits+1)+(n_samples % (S.n_splits + 1)), S.max_train_size)-1:split[1] - S.gap - 1 for (i, split) in enumerate(test_splits)]  
    return collect(zip(train_splits, test_splits))
end


end