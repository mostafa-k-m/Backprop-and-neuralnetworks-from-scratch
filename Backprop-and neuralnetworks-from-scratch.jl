### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ bc348cf3-f6a4-4a8c-8a8e-c48e036882ac
begin
	using Markdown
	using StatsFuns
	using Plots
	using Random
	using LinearAlgebra
end

# ╔═╡ ee90e2d1-3fee-4740-a40a-8faa553e81f5
md"""
# Importing Needed Packages
"""

# ╔═╡ a896bb51-b267-44f2-b2d0-6f5599784df2
md"""
# Building The Neral Network
"""

# ╔═╡ 099a96a4-702b-4bc7-a789-0260fa6a8985
md"""
## Defining a `Weights` Type to hold all Weight Vectors for Convenience
"""

# ╔═╡ 9b612e21-338e-4cba-bb23-75051348e523
mutable struct Weights
	W::Array{Float64}
	V::Array{Float64}
	U::Array{Float64}
end

# ╔═╡ d6875cd4-82cb-4d94-99b0-689d3858fe39
md"""
## Defining The Forward Pass Function
σ (the Sigmoid function) will be used as the activation function for all hidden layers. A linear activation function is used for the output layer 
"""

# ╔═╡ 695e4b82-8a17-4595-93fb-6cd7f5d126bc
σ = StatsFuns.logistic

# ╔═╡ 2ce4c710-4316-4e88-9e2a-58e359b89b41
function forwardProp(z::Float64, weights::Weights)
	x̄ = weights.W * [1,z...]
	x = σ.(x̄)
	pushfirst!(x, 1.)
	ȳ = weights.V * x
	y = σ.(ȳ)
	pushfirst!(y, 1.)
	o = weights.U * y
	return o, y, x
end

# ╔═╡ 1aba7267-1cb8-427a-aec9-cd0bb1850e2b
md"""
## Defining The Back Propagation Function

We have chosen to perform Back Propagation using matrices and linear algebra in this assignment as defined in section `5.2` of the first lecture notes. This choice was made for Three main reasons:
- Lack of comfort with Julia (for loops having a scope of their own is something that I deeply struggled with)
- For consistency (since the forward pass function from the lecture also used linear algebra
- The code was much more intuitive for me once I started using the matrices approach. fewer variables were vague because of notation and I could easily track matrix sizes which lead to me catching errors early on

This approach, I believe, gave me a much deeper understanding of back propagation.
"""

# ╔═╡ fc4a0718-b0aa-4d34-a5e5-dfd6c351cf06
md"""
σ̇ is a function that we defined to caculate the differential of Sigmoid.
"""

# ╔═╡ 3d6042a7-749a-4a21-99b2-83c27864ae74
function σ̇(x)
	σ.(x) .* (1 .- σ.(x))
end

# ╔═╡ e6ce4b37-1087-4a32-a3c3-add7b82b17c5
function backProp(z::Float64, o::Array{Float64}, t::Float64, y::Array{Float64}, x::Array{Float64}, weights::Weights)
	δₒ = o - [t]
	∂E_∂U = δₒ.*transpose(y)
	
	δₕ₂	 = (Diagonal(σ̇(y))*transpose(weights.U)*δₒ)[2:end]
	∂E_∂V = δₕ₂.*transpose(x)

	δₕ₁ = (Diagonal(σ̇(x))*transpose(weights.V)*δₕ₂)[2:end]
	∂E_∂W = δₕ₁.*transpose([1,z...])
	return ∂E_∂W, ∂E_∂V, ∂E_∂U
end

# ╔═╡ 48f7e2ed-15a6-4e50-9b59-0c971c447117
md"""
## Writing Functions to Perform Batch Training for Each Epoch

We started wirh writing a function to train over a batch of a given size. the function will calculate the average gradients over this patch, update the weights and then return them.

The function will also calculate the cost (Quadratic cost) for each time a forward pass is made. The sum of the batch cost is returned
"""

# ╔═╡ 4cebe9a6-3e7a-4e31-893e-5efcdbf0921b
function batchLearn(z::Array{Float64}, t::Array{Float64}, weights::Weights, η::Float64)
	cost = 0
	∂E_∂W = zeros(6, 2)
	∂E_∂V = zeros(3, 7)
	∂E_∂U = zeros(1, 4)
	lenBatch = length(z)
	for i in 1:lenBatch
		oₜₑₘₚ, yₜₑₘₚ, xₜₑₘₚ = forwardProp(z[i], weights)
		∂E_∂Wₜₑₘₚ, ∂E_∂Vₜₑₘₚ, ∂E_∂Uₜₑₘₚ = backProp(z[i], oₜₑₘₚ, t[i], yₜₑₘₚ, xₜₑₘₚ, weights)
		∂E_∂W += ∂E_∂Wₜₑₘₚ
		∂E_∂V += ∂E_∂Vₜₑₘₚ
		∂E_∂U += ∂E_∂Uₜₑₘₚ
		cost += .5 * (oₜₑₘₚ[1] - t[i])^2
	end

	# ∂E_∂W ./= lenBatch
	# ∂E_∂V ./= lenBatch
	# ∂E_∂U ./= lenBatch

	weights.W += -η .* ∂E_∂W
	weights.V += -η .* ∂E_∂V
	weights.U += -η .* ∂E_∂U
	return weights, cost
end

# ╔═╡ 3fca4a1c-407c-498d-ad88-06ee85620ba9
md"""
Next, We defined a function that completes the training over one epoch by calling `batchLearn` iteratively until all the training dataset is exhausted. Each time `batchLearn` is called the `weights` are updated and are passed to the next batch.

`trainOneEpoch` takes a `batchSize` parameter which controls how many datapoints are considered a batch.

Additionally, the sum of cost is divided by the number of datapoints in the training set and returned as the average cost. This is done just to give indication that the network is learning each epoch.
"""

# ╔═╡ a984cc3d-6052-467b-b9de-b63c2b681803
function trainOneEpoch(z::Array{Float64}, t::Array{Float64}, weights::Weights, batchSize::Int64, η::Float64)
	cost = 0
	lenData = length(z)
	curBatchStart = 1
	while curBatchStart < lenData
		batchInputs = z[curBatchStart:curBatchStart + batchSize]
		batchTargets = t[curBatchStart:curBatchStart + batchSize]
		weights, cost = batchLearn(batchInputs, batchTargets, weights, η)
		cost += cost
		curBatchStart += batchSize
		if (curBatchStart + batchSize) > lenData
			batchSize = lenData - curBatchStart
		end
	end
	cost /= lenData
	return weights, cost
end


# ╔═╡ 8d5503a9-3cfc-4ef8-b032-7f49ddffba3d
md"""
Finally, `TrainNN` is a function where we can pass the entire training dataset, the targets and define the number of epochs and the batch size desired. The function will then carry out the training by calling `trainOneEpoch` for the number of epochs defined. After each epoch, the average cose will be printed.
"""

# ╔═╡ 1d57edbf-0ad9-453f-a0b0-953e8b24eb38
function TrainNN(z::Array{Float64}, t::Array{Float64}, noEpochs::Int64, batchSize::Int64, η::Float64)
	weights = Weights(randn(6, 2), randn(3, 7), randn(1, 4))
	for epochNo in 1:noEpochs
		weights, cost = trainOneEpoch(z, t, weights, batchSize, η)
		println("Epoch Number: $epochNo, Total Error: $cost")
	end
	return weights
end

# ╔═╡ f8d91682-4a8d-4777-9894-ff956b444b65
md"""
# Training the Network and Plotting the results

This network takes one input and performs regression to predict the value of one output. the function `f(x) = x^2 + 2x + 1` is used to generate an Ad-hoc training dataset.
"""

# ╔═╡ 5fb5e1c3-a783-469e-b428-80d84cd67528
begin
	Random.seed!(132)
	z = rand(-1:0.01:1, 500)
	f(x) = x^2 + 2x + 1
	t = f.(z)
end

# ╔═╡ 6ba11f4c-991c-4ff9-a6c4-31a4647b2e3e
weights = TrainNN(z, t, 1000, 5, 0.0001)

# ╔═╡ 3dad7af1-6f7e-4ed0-8e92-2e48a4904db5
function predict(z::Vector{Float64}, weights::Weights)
	o = []
	for z_ in z
		x̄ = weights.W * [1,z_...]
		x = σ.(x̄)
		pushfirst!(x, 1.)
		ȳ = weights.V * x
		y = σ.(ȳ)
		pushfirst!(y, 1.)
		push!(o, (weights.U * y)[1])
	end
	return o
end

# ╔═╡ 8611143f-8628-42b2-b995-17c1289a0211
scatter(z, f.(z))

# ╔═╡ 5036127a-5b40-40bb-98ae-e4cafb633978
begin
	zₜₑₛₜ  = rand(-1:0.01:1, 100)
	scatter!(zₜₑₛₜ, predict(zₜₑₛₜ, weights))
end

# ╔═╡ 6d610018-917b-444a-91db-f59b99d133f9
md"""
# Observations
It could be observed from the plot above that the model is fitting the range from (-1 to 1) reasonably well. However this model suffers from overfitting when ranges more broad than this are used. and the model tends to just predict the average y-value for every point on the x-axis.
Below we will try two more ranges one more narrow and one more broad. We will observe that with increased "broadness" the model performs much worse.
"""

# ╔═╡ 2505af0e-8c44-4e65-a377-f04753f04f3d
begin
	begin
		Random.seed!(132)
		z_2 = rand(-1:0.01:.5, 500)
		t_2 = f.(z_2)
	end
	weights2 = TrainNN(z_2, t, 1000, 100, 0.001)
	scatter(z_2, f.(z))
	begin
		zₜₑₛₜ2  = rand(-1:0.01:.5, 100)
		scatter!(zₜₑₛₜ2, predict(zₜₑₛₜ2, weights2))
	end
end

# ╔═╡ cae2b1df-b4c4-4e98-bca1-18fbd339502d
begin
	begin
		Random.seed!(132)
		z_3 = rand(-10:0.01:5, 500)
		t_3 = f.(z_3)
	end
	weights3 = TrainNN(z_3, t, 1000, 100, 0.001)
	scatter(z_3, f.(z))
	begin
		zₜₑₛₜ3  = rand(-10:0.01:5, 100)
		scatter!(zₜₑₛₜ3, predict(zₜₑₛₜ2, weights3))
	end
end

# ╔═╡ Cell order:
# ╠═ee90e2d1-3fee-4740-a40a-8faa553e81f5
# ╠═bc348cf3-f6a4-4a8c-8a8e-c48e036882ac
# ╠═a896bb51-b267-44f2-b2d0-6f5599784df2
# ╠═099a96a4-702b-4bc7-a789-0260fa6a8985
# ╠═9b612e21-338e-4cba-bb23-75051348e523
# ╠═d6875cd4-82cb-4d94-99b0-689d3858fe39
# ╠═695e4b82-8a17-4595-93fb-6cd7f5d126bc
# ╠═2ce4c710-4316-4e88-9e2a-58e359b89b41
# ╠═1aba7267-1cb8-427a-aec9-cd0bb1850e2b
# ╠═fc4a0718-b0aa-4d34-a5e5-dfd6c351cf06
# ╠═3d6042a7-749a-4a21-99b2-83c27864ae74
# ╠═e6ce4b37-1087-4a32-a3c3-add7b82b17c5
# ╠═48f7e2ed-15a6-4e50-9b59-0c971c447117
# ╠═4cebe9a6-3e7a-4e31-893e-5efcdbf0921b
# ╠═3fca4a1c-407c-498d-ad88-06ee85620ba9
# ╠═a984cc3d-6052-467b-b9de-b63c2b681803
# ╠═8d5503a9-3cfc-4ef8-b032-7f49ddffba3d
# ╠═1d57edbf-0ad9-453f-a0b0-953e8b24eb38
# ╠═f8d91682-4a8d-4777-9894-ff956b444b65
# ╠═5fb5e1c3-a783-469e-b428-80d84cd67528
# ╠═6ba11f4c-991c-4ff9-a6c4-31a4647b2e3e
# ╠═3dad7af1-6f7e-4ed0-8e92-2e48a4904db5
# ╠═8611143f-8628-42b2-b995-17c1289a0211
# ╠═5036127a-5b40-40bb-98ae-e4cafb633978
# ╠═6d610018-917b-444a-91db-f59b99d133f9
# ╠═2505af0e-8c44-4e65-a377-f04753f04f3d
# ╠═cae2b1df-b4c4-4e98-bca1-18fbd339502d
