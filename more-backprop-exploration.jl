### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f165a786-956b-11eb-35a9-7d92f08f63f7
begin
	using StatsFuns
	using Plots
	using Random
	using LinearAlgebra
	using PlutoUI
	using BenchmarkTools
	using Flux
end

# ╔═╡ 58752de6-957d-11eb-2edf-83f720e0d88e
md"""
   # Backpropagation in Practice: Doing even more explorations

   Here we will do some exploring based on [our third lecture](https://www.youtube.com/watch?v=r-ww2ie-qms)

   Getting all the Packges we will need
"""


# ╔═╡ ac4fea9e-fd90-4f98-8ea8-ee69a21125a6
md"""
## Code to Build the NN
"""

# ╔═╡ bcd0821b-7d2e-493a-884a-0a9f2f2eb9de
md"""
**We have made a type called Weights which is a mutable struct. This struct is going to store all the weights of our NN.**

**To generalize this solution we can define a function to instantiate the Weights (constructor) object where the user can define the number of desired layers.**

**For this assignment, we are going to hard code the number of weight vectors assuming an NN made of 3 layers (two hidden layers and an output layer)**
"""

# ╔═╡ 5fe25056-3c9b-41aa-a297-163056a7b0bf
mutable struct Weights
	W::Array{Float64}
	V::Array{Float64}
	U::Array{Float64}
end

# ╔═╡ dfd3a5ca-8b61-429e-8a34-bd02bfa0e5bf
md"""
**We defined a function to instantiate the weights (a constructor). the constructor takes two partameters, the number of first hidden layer nodes and the second hidden layer nodes**
"""

# ╔═╡ d7220652-e603-4bde-a936-e05a7781ce73
init_weights(d_h1, d_h2) = Weights(randn(d_h1, 2), randn(d_h2, d_h1 + 1), randn(1, d_h2 + 1))

# ╔═╡ 7ff88140-957f-11eb-182c-3d5bb8773e44
md"We recylce the logistic function for our notation" 

# ╔═╡ 56ecd7aa-956c-11eb-0656-97eece260d17
σ=StatsFuns.logistic

# ╔═╡ 2abff314-962d-11eb-049c-25ba0f188956
σ̇(x)=σ.(x) .* (1 .- σ.(x))

# ╔═╡ 828eb753-b0ed-40b9-b40a-2d8791c07683
md"""
**We modified the forwardProp function to move the loop that iterates over the vector z (or ``x`` in this document) out of the function.**

**This choice was made to allow the author to do some structural changes that they thought made the code more modular and easy to understand**
"""

# ╔═╡ 5e08ecdc-d404-4c63-b1e9-3fc54d1e3819
# Forward propagation 
function forwardProp(z::Float64, weights::Weights)

	x̄ = weights.W * [1,z...]
	x = σ.(x̄)
	pushfirst!(x, 1.)
	ȳ = weights.V * x
	y = σ.(ȳ)
	pushfirst!(y, 1.)
	o = weights.U * y
	
	return o, y, x, ȳ, x̄
end

# ╔═╡ d532d013-0d4a-4236-ac3d-c06b8ddedd67
md"""
**We are going to utilize multiple dispatch to define another forwardProp function that takes a vector of floats for z instead of a float. This function is going to be used to facilitate evaluation later on.**
"""

# ╔═╡ 262d175d-1663-4599-9967-e946a7814213
function forwardProp(z::Vector{Float64}, weights::Weights)
	o = []

	for z_ in z
		o_, _, _, _, _ = forwardProp(z_, weights)
		push!(o, o_[1])
	end

	return o
end

# ╔═╡ c4b8f32d-9985-41ff-b60f-8cf7709cc2b9
md"""
**Our first implementation of ``backProp`` is based on matrix operations instead of using loops to compute the gradients**
"""

# ╔═╡ ea36b410-a49b-42bd-b8a1-24502fbeeba7
function backProp(z::Float64, o::Array{Float64}, t::Float64,
		y::Array{Float64}, x::Array{Float64}, ȳ::Array{Float64},
		x̄::Array{Float64}, weights::Weights)

	δₒ = o - [t]
	∂E_∂U = reshape(δₒ, length(δₒ), 1) * transpose(y)
	
	δₕ₂	 = Diagonal(σ̇(ȳ)) * (transpose(weights.U) * δₒ)[2:end]
	∂E_∂V = reshape(δₕ₂, length(δₕ₂), 1) * transpose(x)

	δₕ₁ = Diagonal(σ̇(x̄)) * (transpose(weights.V) * δₕ₂)[2:end]
	∂E_∂W = reshape(δₕ₁, length(δₕ₁), 1) * transpose([1,z...])

	return ∂E_∂W, ∂E_∂V, ∂E_∂U
end


# ╔═╡ 8c3d0cd0-3a7f-4c55-bdd9-71167ee3a362
md"""
**Our second implementation of ``backProp`` is based on the same matrix operations method but it utilizes julia's ``broadcast`` capabilities.**

**We will call this implementation ``backPropBroadcast``**
"""

# ╔═╡ c78af765-80ab-48df-ac28-de13a3d42035
function backPropBroadcast(z::Float64, o::Array{Float64}, t::Float64,
		y::Array{Float64}, x::Array{Float64}, ȳ::Array{Float64},
		x̄::Array{Float64}, weights::Weights)

	δₒ = o - [t]
	∂E_∂U = δₒ .* transpose(y)

	δₕ₂	 = Diagonal(σ̇(ȳ)) * (transpose(weights.U) * δₒ)[2:end]
	∂E_∂V = δₕ₂ .* transpose(x)

	δₕ₁ = Diagonal(σ̇(x̄)) * (transpose(weights.V) * δₕ₂)[2:end]
	∂E_∂W = δₕ₁ .* transpose([1,z...])

	return ∂E_∂W, ∂E_∂V, ∂E_∂U
end

# ╔═╡ 5e9f88b1-5280-4c2f-88f3-c6ef485db137
md"""
**Our third implementation of ``backProp`` uses loops to compute the gradients with no matrix operations at all.**

**We will call this implementation ``backPropLong``**
"""

# ╔═╡ 80769420-664c-4d3a-bb0c-089f3d2b5ba4
function backPropLong(z::Float64, o::Array{Float64}, t::Float64,
		y::Array{Float64}, x::Array{Float64}, ȳ::Array{Float64},
		x̄::Array{Float64}, weights::Weights)

	δₒ, ∂E_∂U = o - [t], zeros(size(weights.U))
	δₕ₂, ∂E_∂V = zeros(size(weights.V, 1)), zeros(size(weights.V))
	δₕ₁, ∂E_∂W = zeros(size(weights.W, 1)), zeros(size(weights.W))

	for i in 1:size(weights.U,1), j in 1:size(weights.U,2)
		∂E_∂U[i,j] += δₒ[i]*y[j]
	end

	_temp_arg = 0
	for i in 1:size(weights.V,1)
		for j in 1:length(δₒ)
			_temp_arg += δₒ[j]*weights.U[j, i+1]
		end
		δₕ₂[i] += σ̇(ȳ[i])*_temp_arg
	end

	for i in 1:size(weights.V,1), j in 1:size(weights.V,2)
		∂E_∂V[i,j] += δₕ₂[i]*x[j]
	end

	_temp_arg = 0
	for i in 1:size(weights.V,1)
		for j in 1:length(δₕ₂)
			_temp_arg += δₕ₂[j]*weights.V[j, i+1]
		end
		δₕ₁[i] += σ̇(x̄[i])*_temp_arg
	end

	for i in 1:size(weights.W,1), j in 1:size(weights.W,2)
		∂E_∂W[i,j] += δₕ₁[i]*[1,z...][j]
	end

	return ∂E_∂W, ∂E_∂V, ∂E_∂U
end

# ╔═╡ 0c4bbd2c-4d24-47ca-b204-647bf463f729
md"""
## Code to Conduct Training and Evaluation
"""

# ╔═╡ cce31278-c5ef-4b94-ab0a-2853d990d1dc
md"""
**We are going to implement batch learning through our ``batchLearn`` function. We should recall that our weights constructor handled the number of nodes in our two hidden layers. in this function we are going to infer these values from the sizes of our weights, which are parameters to this function.**

**This function also takes ``z`` (training dataset), ``t`` (target dataset), ``bp\_fun`` (our choice of ``backProp`` implementations), and ``η`` (learning rate) as parameters.**

**In this function, we can observe the implications of our choice to modify ``forwardProp`` earlier. It made it possible for us to make this for loop which is going to simply call ``forwardProp`` and then ``bp\_fun`` consecutively on each data point in the batch. and then update the wieghts once for the batch.**

**The change allowed us to encapsulate both the forward and back propagation steps in the same for loop**

**The function calculates the cost which is chosen to be ``MSE`` for this application. Taking into consideration that calculating ``RMSE`` is more computationally expensive and both are equivalent for  our application in practice.**

**The function returns the updated ``weights`` after the batch is done and also the ``cost`` accumulated during the training process.**
"""

# ╔═╡ d6d82a33-698a-4acf-94c5-cda583bd0b9f
MSE(x, y) = (y- x)^2

# ╔═╡ 98d43eda-4bcf-46a3-bc56-5d424dada3cf
function batchLearn(z::Array{Float64}, t::Array{Float64},
		bp_fun::Function, η::Float64, weights::Weights)

	cost = 0
	∂E_∂W = zeros(size(weights.W,1), size(weights.W,2))
	∂E_∂V = zeros(size(weights.V,1), size(weights.V,2))
	∂E_∂U = zeros(size(weights.U,1), size(weights.U,2))
	lenBatch = length(z)

	for i in 1:lenBatch
		oₜₑₘₚ, yₜₑₘₚ, xₜₑₘₚ, ȳₜₑₘₚ, x̄ₜₑₘₚ = forwardProp(z[i], weights)
		∂E_∂Wₜₑₘₚ, ∂E_∂Vₜₑₘₚ, ∂E_∂Uₜₑₘₚ = bp_fun(z[i], oₜₑₘₚ, t[i], yₜₑₘₚ, xₜₑₘₚ, ȳₜₑₘₚ, x̄ₜₑₘₚ, weights)
		∂E_∂W .+= ∂E_∂Wₜₑₘₚ
		∂E_∂V .+= ∂E_∂Vₜₑₘₚ
		∂E_∂U .+= ∂E_∂Uₜₑₘₚ
		cost += MSE(oₜₑₘₚ[1], t[i])
	end

	weights.W -= η .* ∂E_∂W
	weights.V -= η .* ∂E_∂V
	weights.U -= η .* ∂E_∂U

	return weights, cost
end

# ╔═╡ 9f7597fe-448c-4ffc-8a20-dfadf4fb1856
md"""
**Next, we defined ``trainOneEpoch`` to iteratively call ``batchLearn``, after specifying a certain batch size, until the training dataset is exhausted. Therefore, completing an epoch**

**This function takes ``z`` (training dataset), ``t`` (target dataset), ``batchSize`` (batch size), ``bp\_fun`` (our choice of ``backProp`` implementations), ``η`` (learning rate) and ``weights`` as parameters. all of these parameters, except ``batchSize`` are passed `as is` to ``batchLearn``**

**the function returns the updated ``weights`` after the epoch is done and also the ``cost`` accumulated during the training process.**
"""

# ╔═╡ 2f43695b-d665-4ba6-bec7-f3c7428692ee
function trainOneEpoch(z::Array{Float64}, t::Array{Float64},
		batchSize::Int64, bp_fun::Function, η::Float64,
		weights::Weights)

	cost = 0
	lenData = length(z)
	curBatchStart = 1

	curBatchSize = batchSize
	while curBatchStart < lenData
		batchInputs = z[curBatchStart:curBatchStart + curBatchSize]
		batchTargets = t[curBatchStart:curBatchStart + curBatchSize]
		weights, cost = batchLearn(batchInputs, batchTargets, bp_fun, η, weights)
		cost += cost
		curBatchStart += curBatchSize + 1

		if (curBatchStart + batchSize) > lenData
			curBatchSize = lenData - curBatchStart
		end

	end

	cost /= lenData

	return weights, cost
end

# ╔═╡ 96fa06eb-a445-4146-84b1-b09f8708d868
md"""
**Finally, we have defined the ``train`` function which instantiates the weights randomly using our previously defined constructor, Then calls ``trainOneEpoch`` for the defined number of epochs.**

**This function takes ``z`` (training dataset), ``t`` (target dataset), ``noEpochs`` (number of epochs), ``batchSize`` (batch size), ``bp\_fun`` (our choice of ``backProp`` implementations), ``η`` (learning rate), ``d\_h1`` (number of first hidden layer nodes) and ``d\_h2`` (number of second hidden layer nodes)**

"""

# ╔═╡ c15ca29b-c746-49b7-accb-db671110ab71
function train(z::Array{Float64}, t::Array{Float64},
		noEpochs::Int64, batchSize::Int64, bp_fun::Function,
		η::Float64, d_h1::Int64, d_h2::Int64)

	weights::Weights=init_weights(d_h1, d_h2)
	cost_hist = zeros(noEpochs)

    for epochNo in 1:noEpochs
		weights, cost = trainOneEpoch(z, t, batchSize, bp_fun, η, weights)
		cost_hist[epochNo] = cost
		# println("Epoch Number: $epochNo, Total Error: $cost")
	end

	return weights, cost_hist
end

# ╔═╡ eb4c8118-5bd1-498a-8acb-68bfc241d547
md"""
**We again utilize julia's multiple dispatch concept to define two implementations of the ``predict`` function, to predict the corresponding value for either a single datapoint (float) or a dataset (array of floats) depending on the type of evaluation needed.**
"""

# ╔═╡ edce82e8-60b5-4ee0-9ec5-c1aefe78213b
function predict(z::Vector{Float64}, weights::Weights)

	o = forwardProp(z, weights)

	return o
end

# ╔═╡ 0fd1e9b9-8cd3-4e06-8477-911db1d1b726
function predict(z::Float64, weights::Weights)

	o, _, _, _, _ = forwardProp(z, weights)
	return o[1]
end

# ╔═╡ 6b52fa24-4d43-4ae3-9a4d-2a5ed750be6a
md"""
## Code to Plot the Results!
"""

# ╔═╡ 7a15c34a-c2ee-4c86-a1ba-fc8c9a209380
md"""
**We defined the ``addplot!`` which modifies an existing plot in-place and adds a linear plot based on an NN's predictions. Displaying a given label.**
"""

# ╔═╡ 528b7768-efaf-45ae-9006-b767cca476e6
function addplot!(testSet::Vector{Float64}, weights::Weights, label::String)
	plot!(testSet, predict(testSet, weights), label=label)
end

# ╔═╡ ebe10f30-1eac-4153-9c31-50e1d2991bc9
md"""
**We defined the ``initPlot`` Which creates the base plot showing the underlying function and a sample of noisy input that was used for training.**
"""

# ╔═╡ 4cb8c1d9-b78e-40b6-88a9-f9ca3d139913
function initPlot(f::Function, x::Vector{Float64}, t::Vector{Float64}, title::String)
	graph = plot(f, label="Original Function", title=title)
	scatter!(x, t, label="Noisy Input")
	return graph
end

# ╔═╡ 22c5b690-6cc4-4733-8bf5-3127edfe6ebc
md"""
**We defined ``comparisonPlots!``, a function that iteratively calls ``addplot!`` for a number of NNs and displays them with appropriate labels.**
"""

# ╔═╡ c694b303-04da-43fb-bd5d-14c21536f9d2
function comparisonPlots!(f::Function, x::Vector{Float64}, t::Vector{Float64},
		testSet::Vector{Float64}, weightsArray::Vector{Weights},
		labelsArray::Vector{String}, title::String)
	graph = initPlot(f, x, t, title)
	for i in 1:length(weightsArray)
		weights, label = weightsArray[i], labelsArray[i]
		plot!(testSet, predict(testSet, weights), label=label)
	end
	return graph
end

# ╔═╡ 5e9df45f-ba25-46f3-9f3e-2aac6aea0dab
md"""
**We also defined the ``learningPlots`` to plot the learning curves for any number of NNs to compare them.**
"""

# ╔═╡ 0ea44ec1-ac43-4896-a674-26ccd395a86a
function learningPlots(costHistArray::Vector{Vector{Float64}},
		labelsArray::Vector{String}, title::String)
	graph = plot(title=title, xlabel = "Epoch", ylabel = "MSE")
	for i in 1:length(costHistArray)
		costHist, label = costHistArray[i], labelsArray[i]
		plot!(1:length(costHist), costHist, label=label)
	end
	return graph
end

# ╔═╡ edfb42fc-9581-11eb-2751-29d1e5d6179e
md"""
## Assignment Questions 


!!! question "Question 1"

    Implement `backprop!`, `backprop_long!`, `backprop_broadcast!`, and `train!`.
	Show that they work propelry via a plot the showing how they generalize. 

"""

# ╔═╡ 74c0efa4-d795-474c-be72-98c1342ed5c8
md"""
This is the code from the original assignment document to initiate the underlying function the noisy ``t`` dataset.
"""

# ╔═╡ 2f88b974-957e-11eb-0907-87285dc60d9b
md"We define a simple quadratic function $$f(x)$$ that we will use to sample from to train our 
neural network"

# ╔═╡ 820488dc-9562-11eb-3cf1-8feaf98fa4c8
f(x) = x^2 + 2x +1

# ╔═╡ adc72148-9562-11eb-1d32-2fa66216ed52
begin 
   Random.seed!(132)
   x = rand(-5:0.01:5,20)
   t = f.(x) + randn(length(x))
end;  #The semicolum here just supresses the output

# ╔═╡ 624317ba-957e-11eb-292d-319fe8d68146
md"We generate 12 random samples over the range (-5,5), for each sample we get the output of our $$f(x)$$ and add a little bit of Gaussian noise"

# ╔═╡ e202aba5-7e54-4524-965a-c8acf89fb1bf
md"""
**Below we are going to plot the results of three NNs, trained with our three implementations of ``backProp``. These NNs are all going to use the following parameters:**

**No. of Epochs = 1500**

**Batch Size = 3**

**Learning Rate = 0.001**

**No. of 1st hidden layer nodes = 6**

**No. of 2nd hidden layer nodes = 3**
"""

# ╔═╡ 13fda10a-d15d-44d3-b4bc-7f00141ec0a0
begin
	xₜₑₛₜ_  = sort(rand(-5:0.01:5,100))
	weights_matrix_test, cost_history_matrix_test = train(x, t, 1500, 5,
			backProp, .001, 6, 3)
	weights_broadcast_test, cost_history_broadcast_test = train(x, t, 1500, 5,
			backPropBroadcast, .001, 6, 3)
	weights_long_test, cost_history_long_test = train(x, t, 1500, 5,
			backPropLong, .001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_test, weights_broadcast_test, weights_long_test],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "Showing the different backProp implementations")
end

# ╔═╡ 2be535da-ae7c-4914-99ee-72b26ee3a907
md"""
**Below we can see the learning curves are very similar. This proves that our implementation of the three ``backProp`` variants was correct**
"""

# ╔═╡ a3065f97-0d9d-4d18-ad61-2da94c813fd3
learningPlots([cost_history_matrix_test, cost_history_broadcast_test, cost_history_long_test],
			["Matrix NN loss", "Broadcast NN loss", "Long (Loop) NN loss"], "Learning Curves")

# ╔═╡ 045260ba-62e0-407a-8485-0c9ef6cc631c
md"""
!!! question "Question 2"
    Compare the  accuracy `backprop!`, `backprop_long!`, `backprop_broadcast!`
	on *test data* as 

		1. The number of layer 2 (dh_1) hidden node
		2. The learning rate changes
		3. The number of epocsh increase 
"""

# ╔═╡ 7f03c154-4ecb-4a17-be0d-5c58a0b43bb3
md"""
**To explore the effect of multiple parameters on how our NN learns we have made sliders for each of them. The sliders immediately change the plot below so we can see the result of the experiment.**
"""

# ╔═╡ 5fa11ae4-8cea-49af-a1ec-625290e36199
begin
	d_h1_s=@bind d_h1 Slider(4:10, default=6)
	d_h2_s=@bind d_h2 Slider(2:10, default=4)
	noEpochs_s=@bind noEpochs Slider(10:2000, default=500)
	batchSize_s=@bind batchSize Slider(2:10, default=5)
	η_s=@bind η_FROM_SLIDER Slider(1:100, default=5)
	md"""
	Hidden layer one nodes: $d_h1_s
	
	Hidden layer two nodes: $d_h2_s
	
	No of Epochs: $noEpochs_s
	
	Batch Size: $batchSize_s
	
	value for η: $η_s
	
	"""
end

# ╔═╡ c2d43c2b-6bc7-48fc-b49d-d5a275af14c1
begin
		η = η_FROM_SLIDER/1000
		"d_h1:$d_h1", "d_h2:$d_h2", "noEpochs:$noEpochs", "batchSize:$batchSize", "η:$η"
end

# ╔═╡ 194a24b9-04ab-4101-bccb-a8a5ae7c51ad
begin
	xₜₑₛₜ  = sort(rand(-5:0.01:5,100))
	weights_matrix, cost_hist_matrix = train(x, t, noEpochs, batchSize,
			backProp, η, d_h1, d_h2)
	weights_broadcast, cost_hist_broadcast = train(x, t, noEpochs, batchSize,
			backPropBroadcast, η, d_h1, d_h2)
	weights_long, cost_hist_long = train(x, t, noEpochs, batchSize,
			backPropLong, η, d_h1, d_h2)
	comparisonPlots!(f, x, t, xₜₑₛₜ,
			[weights_matrix, weights_broadcast, weights_long],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "Interactive Plot")
end

# ╔═╡ 82e056a9-5294-40be-81ab-a07031e8be44
md"""
**Calculating and printing MSE (our measure of accuracy) for each of our three networks**
"""

# ╔═╡ 65f196b5-ae2c-4e2a-baaf-cc1c8d8949dd
function calcAccuracy(testSet, weights, target)
	error = 0
	for testPoint in testSet
		error += MSE(testPoint, predict(testPoint, weights))
	end
	return error/length(testSet)
end

# ╔═╡ 5a2e7072-fd8c-4e72-92a4-7714b3a0c8c5
begin
	NN_matrix_accuracy = round(calcAccuracy(x, weights_matrix, t), digits=2)
	NN_broadcast_accuracy = round(calcAccuracy(x, weights_broadcast, t), digits=2)
	NN_long_accuracy = round(calcAccuracy(x, weights_long, t), digits=2)
"NN_matrix_accuracy = $NN_matrix_accuracy", "NN_broadcast_accuracy = $NN_broadcast_accuracy", "NN_long_accuracy = $NN_long_accuracy"
end

# ╔═╡ e6c02565-1086-490b-bc59-ac2fc946f3fc
learningPlots([cost_hist_matrix, cost_hist_broadcast, cost_hist_long],
			["Matrix NN loss", "Broadcast NN loss", "Long (Loop) NN loss"], "Learning Curves")

# ╔═╡ ae72ee52-666a-4b67-b709-2a066f717d85
md"""
!!! question "Question 3"
	How can you explain the results observed in light of all you learned in the course so far. 
"""

# ╔═╡ 3743d3b4-45b3-4dec-a3e4-a5aa48e6179b
md"""
**To answer this question we are going to perform three experiments. We are going to start with a base configuration for an NN as follows:**
- No of Epochs = 1000
- batch size = 5
- learning rate = 0.001
- number of first hidden layer nodes = 6
- number of second hidden layer nodes = 3

**In each experiment, We are going to change only one of the No. of Epochs, the learning rate or the number of second hidden layer nodes parameters, and observe the effect of that change on the fit using the plots that are generated**

**We will also compare the learning curves for each experiment. but we are only going to show the learning curves for the matrix implementation. Since all implementations are similar and we are only interested in studying the effect of change in some hyperparameters as opposed to the NN ``backProp`` implementation**
"""

# ╔═╡ 2e6875ea-1cf0-4622-b1c2-72dc0b67ca0e
md"""
### Effect of Changes in the Number Second Hidden Layer Nodes on the Fit
"""

# ╔═╡ e43b9002-4287-4f61-ade1-eb21ad814319
md"""
During this experiment, we observe that with increasing dh_2, the resulting curve fits the noisy sample more and more. This is to be expected, because with increased model complexity, the model is able distinguish more nuanced features and combinations there of.

An additional explaination to why the fit was best in the first instance of this experiment is that it is considered best practice to have the same number of nodes across layers in shallow NNs. This condition happened to be true in the first instance.

We also notice that the model with the highest number of nodes was overfitting the noisy sample, showing less ability to generalize.
"""

# ╔═╡ 14e74118-325f-4959-a946-4ad307a29879
begin
	weights_matrix_dh2_1, cost_matrix_dh2_1 = train(x, t, 1000, 5,
			backProp, .001, 6, 6)
	weights_broadcast_dh2_1, _ = train(x, t, 1000, 5,
			backPropBroadcast, .001, 6, 6)
	weights_long_dh2_1, _ = train(x, t, 1000, 5,
			backPropLong, .001, 6, 6)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_dh2_1, weights_broadcast_dh2_1, weights_long_dh2_1],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "# of second hidden layer nodes = 6")
end

# ╔═╡ 53dbcfc2-02c3-4a0b-9ed1-51d305e168a3
begin
	weights_matrix_dh2_2, cost_matrix_dh2_2 = train(x, t, 1000, 5,
			backProp, .001, 6, 3)
	weights_broadcast_dh2_2, _ = train(x, t, 1000, 5,
			backPropBroadcast, .001, 6, 3)
	weights_long_dh2_2, _ = train(x, t, 1000, 5,
			backPropLong, .001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_dh2_2, weights_broadcast_dh2_2, weights_long_dh2_2],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "# of second hidden layer nodes = 3")
end

# ╔═╡ 209be3b8-27ea-4283-a7e5-93556dc208f1
begin
	weights_matrix_dh2_3, cost_matrix_dh2_3 = train(x, t, 1000, 5,
			backProp, .001, 6, 1)
	weights_broadcast_dh2_3, _ = train(x, t, 1000, 5,
			backPropBroadcast, .001, 6, 1)
	weights_long_dh2_3, _ = train(x, t, 1000, 5,
			backPropLong, .001, 6, 1)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_dh2_3, weights_broadcast_dh2_3, weights_long_dh2_3],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "# of second hidden layer nodes = 1")
end

# ╔═╡ f8a8ab14-5510-44bf-b870-18eee229aa25
learningPlots([cost_matrix_dh2_1, cost_matrix_dh2_2, cost_matrix_dh2_3],
			["Matrix NN dh2=6", "Matrix NN dh2=3", "Matrix NN dh2=1"], "Learning Curves")

# ╔═╡ 69ae765f-e184-42e1-91db-24ac8bb8540e
md"""
### Effect of Learning Rate Changes on the Fit
"""

# ╔═╡ a4de54ba-32fc-4dae-a4b0-8347c89c3874
md"""
The learning rate is fundamently important to the NNs learning performance as the learning rate dictates the magnitude of update (size of step) that affects the weights after each batch, in the case of batch training.

If the learning rate is too high the updates will be too large that they will overshoot the minimas and fail to converge. If the learning rate is too small, the NN will take a very long time to converge and might be stuck in a local minima. This risk is especially dangerous here as we are using non-adaptive learning rates.

However if the learning rate is reasonable, that will give the NN the best chance of convergence. These three cases, we believe, are nicely displayed in the three examples below.
"""

# ╔═╡ 01ccc3cb-f6d5-4bf4-9782-95976b46e1a1
begin
	weights_matrix_eta_1, cost_matrix_eta_1 = train(x, t, 1000, 5,
			backProp, .1, 6, 3)
	weights_broadcast_eta_1, _ = train(x, t, 1000, 5,
			backPropBroadcast, .1, 6, 3)
	weights_long_eta_1, _ = train(x, t, 1000, 5,
			backPropLong, .1, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_eta_1, weights_broadcast_eta_1, weights_long_eta_1],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "η = .1")
end

# ╔═╡ b00fda49-6080-404a-ab68-9af5b88c94db
begin
	weights_matrix_eta_2, cost_matrix_eta_2 = train(x, t, 1000, 5,
			backProp, .01, 6, 3)
	weights_broadcast_eta_2, _ = train(x, t, 1000, 5,
			backPropBroadcast, .01, 6, 3)
	weights_long_eta_2, _ = train(x, t, 1000, 5,
			backPropLong, .01, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_eta_2, weights_broadcast_eta_2, weights_long_eta_2],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "η = .01")
end

# ╔═╡ b984b122-6e0b-4f4c-ac3a-41cd74eadb14
begin
	weights_matrix_eta_3, cost_matrix_eta_3 = train(x, t, 1000, 5,
			backProp, .001, 6, 3)
	weights_broadcast_eta_3, _ = train(x, t, 1000, 5,
			backPropBroadcast, .001, 6, 3)
	weights_long_eta_3, _ = train(x, t, 1000, 5,
			backPropLong, .001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_eta_3, weights_broadcast_eta_3, weights_long_eta_3],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "η = .001")
end

# ╔═╡ b698ed4f-fe26-42f8-b1fe-0c11c2a96059
learningPlots([cost_matrix_eta_1, cost_matrix_eta_2, cost_matrix_eta_3],
			["Matrix NN η=0.1", "Matrix NN η=0.01", "Matrix NN η=0.001"], "Learning Curves")

# ╔═╡ 451cd040-2714-4f30-bf3f-4b0fe5c4b387
md"""
### Effect of the Number of Epochs Changes on the Fit
"""

# ╔═╡ 0c209c4f-8ca2-4095-b465-35630e8b0526
md"""
The number of epochs is basically the number of times our NN is going to complete a pass over all datapoints in the training dataset. More passes will result in the NN fitting the training dataset more and more closely, provided all other hyper parameters are adequate.

In this experiment, we know that the learning rate is too low. but that's only going to accentuate the effect of the change in the number of epochs.

If the NN does not get exposed to the training set enough the fit is going to be very poor. and if the NN is overtrained on the same dataset it is going to suffer from overfitting and failure to generalize. We can see examples of these phenomena below.
"""

# ╔═╡ 43bdb72f-6667-4614-be64-448a7927e3b6
begin
	weights_matrix_epoch_1, cost_matrix_epoch_1 = train(x, t, 5000, 5,
			backProp, .001, 6, 3)
	weights_broadcast_epoch_1, _ = train(x, t, 5000, 5,
			backPropBroadcast, .001, 6, 3)
	weights_long_epoch_1, _ = train(x, t, 5000, 5,
			backPropLong, .001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_epoch_1, weights_broadcast_epoch_1, weights_long_epoch_1],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "# of Epochs = 5000")
end

# ╔═╡ 23fe6189-71d3-4d88-b92d-29d5f9d7b243
begin
	weights_matrix_epoch_2, cost_matrix_epoch_2 = train(x, t, 2000, 5,
			backProp, .001, 6, 3)
	weights_broadcast_epoch_2, _ = train(x, t, 2000, 5,
			backPropBroadcast, .001, 6, 3)
	weights_long_epoch_2, _ = train(x, t, 2000, 5,
			backPropLong, .001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_epoch_2, weights_broadcast_epoch_2, weights_long_epoch_2],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "# of Epochs = 2000")
end

# ╔═╡ eb6fefbc-eeea-42cc-a7eb-da2610840af6
begin
	weights_matrix_epoch_3, cost_matrix_epoch_3 = train(x, t, 500, 5,
			backProp, .001, 6, 3)
	weights_broadcast_epoch_3, _ = train(x, t, 500, 5,
			backPropBroadcast, .001, 6, 3)
	weights_long_epoch_3, _ = train(x, t, 500, 5,
			backPropLong, .001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_epoch_3, weights_broadcast_epoch_3, weights_long_epoch_3],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "# of Epochs = 500")
end

# ╔═╡ 2dbbf968-a504-4539-b455-9a89dde7191d
begin
	weights_matrix_epoch_4, cost_matrix_epoch_4 = train(x, t, 100, 5,
			backProp, .001, 6, 3)
	weights_broadcast_epoch_4, _ = train(x, t, 100, 5,
			backPropBroadcast, .001, 6, 3)
	weights_long_epoch_4, _ = train(x, t, 100, 5,
			backPropLong, .001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_epoch_4, weights_broadcast_epoch_4, weights_long_epoch_4],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN"], "# of Epochs = 100")
end

# ╔═╡ 7e1841bc-d472-4c9a-89a6-6b4c33785d57
learningPlots([cost_matrix_epoch_1, cost_matrix_epoch_2, cost_matrix_epoch_3, cost_matrix_epoch_4],
			["Matrix NN epohs=5000", "Matrix NN epohs=2000", "Matrix NN epohs=500", "Matrix NN epohs=100"], "Learning Curves")

# ╔═╡ c447a6b4-e5bd-453f-85d3-a9069cb60f8c
md"""
!!! question "Bonus question"
	Working with the (a maybe modified) loss function, use Flux to [take gradiants automatically](https://fluxml.ai/Flux.jl/stable/models/basics/) to do the training. Are you getting the same/similar results compared to the hand crafted version?
"""

# ╔═╡ 2cccc1e7-0f10-4283-9bee-9194a9dd0755
md"""
**We have to refactor our NN generating code to utilize Flux.gradient. We will start by modifying ``forwarProp``, because Flux.gradient does not support mutable arrays in the differentiable function.**

**All of our modified functions for Flus.gradient will have the FG suffix in the function's name.**
"""

# ╔═╡ 27ee010b-6fba-40d5-a287-366ab57a0f6f
function forwardPropFG(z::Float64, weights::Weights)
	
	x̄ = weights.W * [1.,z...]
	x = σ.(x̄)
	ȳ = weights.V * [1.,x...]
	y = σ.(ȳ)
	o = weights.U * [1.,y...]

	return o[1]
end

# ╔═╡ aa581a16-f809-42ea-ae44-158b051d2a82
md"""
**We will define a loss function that calls the modified ``forwardPropFG``. The loss function uses the same MSE concept that we used in the previous NNs. Ha"ving the same loss function makes it easy to compare the learning curves across all of our implementations.**
"""

# ╔═╡ 160af8fe-7b2c-4fdd-9fde-c192aa484ec9
function costFuncFG(z::Float64, t::Float64, weights::Weights)
	return (forwardPropFG(z, weights) - t)^2
end

# ╔═╡ 750d43ce-0780-4044-b62d-67148ef50cda
md"""
**We made a function called ``backPropFG`` which uses Flux.gradient to calclate $$\frac{∂E}{∂U}$$, $$\frac{∂E}{∂V}$$, and $$\frac{∂E}{∂W}$$ by performing autodiff on the loss function with respect to each of the weight vectors.**
"""

# ╔═╡ 67b75175-edc1-4d5c-af18-768f1e71c931
function backPropFG(z::Float64, t::Float64, weights::Weights)
	gs = gradient(() -> costFuncFG(z, t, weights), Flux.params(weights.W, weights.V, weights.U))
	∂E_∂U = gs[weights.U]
	∂E_∂V = gs[weights.V]
	∂E_∂W = gs[weights.W]
	return ∂E_∂U, ∂E_∂V, ∂E_∂W
end

# ╔═╡ e11b30c4-294d-420c-b330-28b8076ab079
md"""
**Finally, we have refactored our training code to call our ``FG`` functions. We have chosen not to modify the previous implementation of the training code and instead make new ``FG`` ones for the sake of simplicity** 
"""

# ╔═╡ c5367f2c-bf17-4ca6-9740-099ca947127d
function batchLearnFG(z::Array{Float64}, t::Array{Float64},
		η::Float64, weights::Weights)

	cost = 0
	∂E_∂W = zeros(size(weights.W,1), size(weights.W,2))
	∂E_∂V = zeros(size(weights.V,1), size(weights.V,2))
	∂E_∂U = zeros(size(weights.U,1), size(weights.U,2))
	lenBatch = length(z)

	for i in 1:lenBatch
		cost = costFuncFG(z[i], t[i], weights)
		∂E_∂U, ∂E_∂V, ∂E_∂W = backPropFG(z[i], t[i], weights)
	end

	weights.W -= η .* ∂E_∂W
	weights.V -= η .* ∂E_∂V
	weights.U -= η .* ∂E_∂U

	return weights, cost
end

# ╔═╡ 46759927-b7a5-4de7-a6d7-edfa953f6a8b
function trainOneEpochFG(z::Array{Float64}, t::Array{Float64},
		batchSize::Int64, η::Float64, weights::Weights)

	cost = 0
	lenData = length(z)
	curBatchStart = 1

	curBatchSize = batchSize
	while curBatchStart < lenData
		batchInputs = z[curBatchStart:curBatchStart + curBatchSize]
		batchTargets = t[curBatchStart:curBatchStart + curBatchSize]
		weights, cost = batchLearnFG(batchInputs, batchTargets, η, weights)
		cost += cost
		curBatchStart += curBatchSize + 1

		if (curBatchStart + batchSize) > lenData
			curBatchSize = lenData - curBatchStart
		end

	end

	cost /= lenData

	return weights, cost
end

# ╔═╡ de85c5e1-1cd5-43c1-a609-5b0dc951425f
function trainFluxGradient(z::Array{Float64}, t::Array{Float64},
		noEpochs::Int64, batchSize::Int64,η::Float64,
		d_h1::Int64, d_h2::Int64)

	weights::Weights=init_weights(d_h1, d_h2)
	cost_hist = zeros(noEpochs)

    for epochNo in 1:noEpochs
		weights, cost = trainOneEpochFG(z, t, batchSize, η, weights)
		cost_hist[epochNo] = cost
		# println("Epoch Number: $epochNo, Total Error: $cost")
	end

	return weights, cost_hist
end

# ╔═╡ 086299c0-23f6-438f-aad9-4a3d5f6155cd
md"""
**To compare the Flux.gradient implementation with our vanilla implementations, we are going to train a NN using trainFluxGradient giving it the same hyper parameters we have used in the test cases before:**

**No. of Epochs = 1500**

**Batch Size = 3**

**Learning Rate = 0.001**

**No. of 1st hidden layer nodes = 6**

**No. of 2nd hidden layer nodes = 3**

**Afterwards, we will plot the predictions of each implementation to judge the fit and will also plot the learning curves of each implementation**
"""

# ╔═╡ 7106f757-b1f5-4fc7-9d79-c213a3fa1408
begin
	weights_FG_test, cost_history_FG_test = trainFluxGradient(x, t, 1500, 5,
			.001, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_matrix_test, weights_broadcast_test, weights_long_test, weights_FG_test],
			["Matrix NN", "Broadcast NN", "Long (Loop) NN", "Flux Gradient NN"], "Our vanilla backProp to backProp using Flux.gradient")
end

# ╔═╡ 900ee17b-f3fc-48c6-abe9-0ad8bb57ef93
md"""
**As could be seen from the plot above, it seems that the Flux.gradient implementation failed to converge given the same test data and the same hyper parameters as our vanilla implementations**

**It should also be noted that looking at the learning curve below, perhaps the Flux.gradient model would benefit from a higher learning rate.** 
"""

# ╔═╡ 620c5940-f7fa-4790-87e2-dda0c1e48ddc
learningPlots([cost_history_matrix_test, cost_history_broadcast_test, cost_history_long_test, cost_history_FG_test],
			["Matrix NN loss", "Broadcast NN loss", "Long (Loop) NN loss", "Flux Gradient NN loss"], "Learning Curves")

# ╔═╡ f6eb81db-c68c-45bd-9f08-332bd6829973
md"""
**In the following experiment we have tried increasing the FG implementation's learning rate. By looking at the learning curves, this helped the model converge in much fewer iterations. However, the goodness of fit is not much improved. we can see the model still underfitting our data and struggling to fit the datapoints close to the boundaries of the dataset's range (they could be considered outliers since our data contains a big cluster of points in the middle with much fewer points close the boundaries).**

**This dataset exhibits heteroscedasticity, and with such a small sample, it is hard to perform regression effectively.**
"""

# ╔═╡ 344cc332-2b23-4b3a-abbe-1826d25e09b1
begin
	weights_FG_test_2, cost_history_FG_test_2 = trainFluxGradient(x, t, 1500, 5,
			.01, 6, 3)
	weights_FG_test_3, cost_history_FG_test_3 = trainFluxGradient(x, t, 1500, 5,
		.05, 6, 3)
	weights_FG_test_4, cost_history_FG_test_4 = trainFluxGradient(x, t, 1500, 5,
		.1, 6, 3)
	comparisonPlots!(f, x, t, xₜₑₛₜ_,
			[weights_FG_test_2, weights_FG_test_3, weights_FG_test_4],
			["FG with η=0.01", "FG with η=0.05", "FG with η=0.1"], "Flux.gradient implementation with higher learning rates")
end

# ╔═╡ f3b8c2b4-bd8f-4e14-9013-09e606b4287c
learningPlots([cost_history_FG_test_2, cost_history_FG_test_3, cost_history_FG_test_4],
			["FG with η=0.01", "FG with η=0.05", "FG with η=0.1"], "Learning Curves")

# ╔═╡ Cell order:
# ╟─58752de6-957d-11eb-2edf-83f720e0d88e
# ╠═f165a786-956b-11eb-35a9-7d92f08f63f7
# ╟─ac4fea9e-fd90-4f98-8ea8-ee69a21125a6
# ╟─bcd0821b-7d2e-493a-884a-0a9f2f2eb9de
# ╠═5fe25056-3c9b-41aa-a297-163056a7b0bf
# ╟─dfd3a5ca-8b61-429e-8a34-bd02bfa0e5bf
# ╠═d7220652-e603-4bde-a936-e05a7781ce73
# ╟─7ff88140-957f-11eb-182c-3d5bb8773e44
# ╠═56ecd7aa-956c-11eb-0656-97eece260d17
# ╠═2abff314-962d-11eb-049c-25ba0f188956
# ╟─828eb753-b0ed-40b9-b40a-2d8791c07683
# ╠═5e08ecdc-d404-4c63-b1e9-3fc54d1e3819
# ╟─d532d013-0d4a-4236-ac3d-c06b8ddedd67
# ╠═262d175d-1663-4599-9967-e946a7814213
# ╟─c4b8f32d-9985-41ff-b60f-8cf7709cc2b9
# ╠═ea36b410-a49b-42bd-b8a1-24502fbeeba7
# ╟─8c3d0cd0-3a7f-4c55-bdd9-71167ee3a362
# ╠═c78af765-80ab-48df-ac28-de13a3d42035
# ╟─5e9f88b1-5280-4c2f-88f3-c6ef485db137
# ╠═80769420-664c-4d3a-bb0c-089f3d2b5ba4
# ╟─0c4bbd2c-4d24-47ca-b204-647bf463f729
# ╟─cce31278-c5ef-4b94-ab0a-2853d990d1dc
# ╠═d6d82a33-698a-4acf-94c5-cda583bd0b9f
# ╠═98d43eda-4bcf-46a3-bc56-5d424dada3cf
# ╟─9f7597fe-448c-4ffc-8a20-dfadf4fb1856
# ╠═2f43695b-d665-4ba6-bec7-f3c7428692ee
# ╟─96fa06eb-a445-4146-84b1-b09f8708d868
# ╠═c15ca29b-c746-49b7-accb-db671110ab71
# ╟─eb4c8118-5bd1-498a-8acb-68bfc241d547
# ╠═edce82e8-60b5-4ee0-9ec5-c1aefe78213b
# ╠═0fd1e9b9-8cd3-4e06-8477-911db1d1b726
# ╟─6b52fa24-4d43-4ae3-9a4d-2a5ed750be6a
# ╟─7a15c34a-c2ee-4c86-a1ba-fc8c9a209380
# ╠═528b7768-efaf-45ae-9006-b767cca476e6
# ╟─ebe10f30-1eac-4153-9c31-50e1d2991bc9
# ╠═4cb8c1d9-b78e-40b6-88a9-f9ca3d139913
# ╟─22c5b690-6cc4-4733-8bf5-3127edfe6ebc
# ╠═c694b303-04da-43fb-bd5d-14c21536f9d2
# ╟─5e9df45f-ba25-46f3-9f3e-2aac6aea0dab
# ╠═0ea44ec1-ac43-4896-a674-26ccd395a86a
# ╟─edfb42fc-9581-11eb-2751-29d1e5d6179e
# ╟─74c0efa4-d795-474c-be72-98c1342ed5c8
# ╠═adc72148-9562-11eb-1d32-2fa66216ed52
# ╟─2f88b974-957e-11eb-0907-87285dc60d9b
# ╠═820488dc-9562-11eb-3cf1-8feaf98fa4c8
# ╟─624317ba-957e-11eb-292d-319fe8d68146
# ╟─e202aba5-7e54-4524-965a-c8acf89fb1bf
# ╠═13fda10a-d15d-44d3-b4bc-7f00141ec0a0
# ╟─2be535da-ae7c-4914-99ee-72b26ee3a907
# ╠═a3065f97-0d9d-4d18-ad61-2da94c813fd3
# ╟─045260ba-62e0-407a-8485-0c9ef6cc631c
# ╟─7f03c154-4ecb-4a17-be0d-5c58a0b43bb3
# ╟─5fa11ae4-8cea-49af-a1ec-625290e36199
# ╟─c2d43c2b-6bc7-48fc-b49d-d5a275af14c1
# ╠═194a24b9-04ab-4101-bccb-a8a5ae7c51ad
# ╟─82e056a9-5294-40be-81ab-a07031e8be44
# ╠═65f196b5-ae2c-4e2a-baaf-cc1c8d8949dd
# ╟─5a2e7072-fd8c-4e72-92a4-7714b3a0c8c5
# ╠═e6c02565-1086-490b-bc59-ac2fc946f3fc
# ╟─ae72ee52-666a-4b67-b709-2a066f717d85
# ╟─3743d3b4-45b3-4dec-a3e4-a5aa48e6179b
# ╟─2e6875ea-1cf0-4622-b1c2-72dc0b67ca0e
# ╟─e43b9002-4287-4f61-ade1-eb21ad814319
# ╠═14e74118-325f-4959-a946-4ad307a29879
# ╠═53dbcfc2-02c3-4a0b-9ed1-51d305e168a3
# ╠═209be3b8-27ea-4283-a7e5-93556dc208f1
# ╠═f8a8ab14-5510-44bf-b870-18eee229aa25
# ╟─69ae765f-e184-42e1-91db-24ac8bb8540e
# ╟─a4de54ba-32fc-4dae-a4b0-8347c89c3874
# ╠═01ccc3cb-f6d5-4bf4-9782-95976b46e1a1
# ╠═b00fda49-6080-404a-ab68-9af5b88c94db
# ╠═b984b122-6e0b-4f4c-ac3a-41cd74eadb14
# ╠═b698ed4f-fe26-42f8-b1fe-0c11c2a96059
# ╟─451cd040-2714-4f30-bf3f-4b0fe5c4b387
# ╟─0c209c4f-8ca2-4095-b465-35630e8b0526
# ╠═43bdb72f-6667-4614-be64-448a7927e3b6
# ╠═23fe6189-71d3-4d88-b92d-29d5f9d7b243
# ╠═eb6fefbc-eeea-42cc-a7eb-da2610840af6
# ╠═2dbbf968-a504-4539-b455-9a89dde7191d
# ╠═7e1841bc-d472-4c9a-89a6-6b4c33785d57
# ╟─c447a6b4-e5bd-453f-85d3-a9069cb60f8c
# ╟─2cccc1e7-0f10-4283-9bee-9194a9dd0755
# ╠═27ee010b-6fba-40d5-a287-366ab57a0f6f
# ╟─aa581a16-f809-42ea-ae44-158b051d2a82
# ╠═160af8fe-7b2c-4fdd-9fde-c192aa484ec9
# ╟─750d43ce-0780-4044-b62d-67148ef50cda
# ╠═67b75175-edc1-4d5c-af18-768f1e71c931
# ╟─e11b30c4-294d-420c-b330-28b8076ab079
# ╠═c5367f2c-bf17-4ca6-9740-099ca947127d
# ╠═46759927-b7a5-4de7-a6d7-edfa953f6a8b
# ╠═de85c5e1-1cd5-43c1-a609-5b0dc951425f
# ╟─086299c0-23f6-438f-aad9-4a3d5f6155cd
# ╠═7106f757-b1f5-4fc7-9d79-c213a3fa1408
# ╟─900ee17b-f3fc-48c6-abe9-0ad8bb57ef93
# ╠═620c5940-f7fa-4790-87e2-dda0c1e48ddc
# ╟─f6eb81db-c68c-45bd-9f08-332bd6829973
# ╠═344cc332-2b23-4b3a-abbe-1826d25e09b1
# ╠═f3b8c2b4-bd8f-4e14-9013-09e606b4287c
