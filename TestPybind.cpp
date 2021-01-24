// TestPybind : Test de fonctionnement de l'embed Python dans C++
// Par Pierre Cuquel

#include "TestPybind.h"
#include <fstream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;
using namespace py::literals;

typedef std::vector<std::pair<float, float>> samples_t;

py::array_t<float> toNumpy(samples_t distrib) {
	py::array_t<float> samples{ static_cast<int>(distrib.size() * 2), (float*)distrib.data() };
	samples.resize({ static_cast<int>(distrib.size()), 2 });
	return samples;
}

samples_t fromNumpy(py::array_t<float> samples) {
	samples_t distrib;
	distrib.reserve(samples.size()); // à voir si c'est vraiment la bonne taille ou s'il faut faire *2
	for (auto& col : samples) {
		std::vector<float> tmp;
		for (auto& item : col)
			tmp.push_back(item.cast<float>());
		distrib.push_back({ tmp[0], tmp[1] });
	}
	return distrib;
}

class RealNVPwrapper {
public:
	RealNVPwrapper(int sampleDims = 2, int hiddenDims = 256, int couplingLayers = 8, float learningRate = 1e-4f, bool allowGPU = true) :
		scope(py::module::import("__main__").attr("__dict__")) // Je comprends pas encore exactement ce que ça fait
	{
		try {
			scope["sample_dims"] = sampleDims;
			scope["hidden_dims"] = hiddenDims;
			scope["coupling_layers"] = couplingLayers;
			scope["learning_rate"] = learningRate;
			scope["allow_gpu"] = allowGPU;
			py::eval_file("../../../pysrc/tutoRealNVP.py", scope);
		}
		catch (py::error_already_set& err) {
			std::cerr << "Exception Python levée :" << std::endl;
			std::cerr << err.what() << std::endl;
		}
	}

	void train(int batchSize = 256, int epochs = 20, float validationSplit = 0.2f) {
		try {
			py::eval("model.fit(normalized_data, verbose=2, batch_size=" + std::to_string(batchSize)
				+ ", epochs=" + std::to_string(epochs)
				+ ", validation_split=" + std::to_string(validationSplit)
				+ ")", scope);
		}
		catch (py::error_already_set& err) {
			std::cerr << "Exception Python levée :" << std::endl;
			std::cerr << err.what() << std::endl;
		}
	}

	samples_t forward(samples_t& samples) {
		try {
			scope["samples"] = toNumpy(samples);
			auto warped = py::eval("model.forward(samples)", scope);
			return fromNumpy(warped);
		}
		catch (py::error_already_set& err) {
			std::cerr << "Exception Python levée :" << std::endl;
			std::cerr << err.what() << std::endl;
		}
		return {};
	}

	samples_t inverse(samples_t& samples) {
		try {
			scope["samples"] = toNumpy(samples);
			auto warped = py::eval("model.inverse(samples)", scope);
			return fromNumpy(warped);
		}
		catch (py::error_already_set& err) {
			std::cerr << "Exception Python levée :" << std::endl;
			std::cerr << err.what() << std::endl;
		}
		return {};
	}

private:
	py::scoped_interpreter guard; // RAII durée de vie de l'interpréteur
	py::dict scope;
};

class UniformDistribution {
public:
	UniformDistribution(float low, float high) :
		dist(low, high)
	{}

	samples_t generate(int size) {
		samples_t distrib;
		for (int i = 0; i < size; i++)
			distrib.push_back({ dist(gen), dist(gen) });
		return distrib;
	}

private:
	std::default_random_engine gen;
	std::uniform_real_distribution<float> dist;
};

int main()
{
	std::cout << "Step 0 : Loading Tensorflow and RealNVP" << std::endl;
	RealNVPwrapper realNVP{ 2, 256, 8, 1e-4, false };

	std::cout << "Step 1 : Training RealNVP" << std::endl;
	realNVP.train();

	std::cout << "Step 2 : Loading an example Numpy array" << std::endl;
	UniformDistribution dist{ 0, 1 };
	samples_t samples = dist.generate(200);

	std::cout << "Step 3 : Warping the samples using RealNVP and catching the result" << std::endl;
	auto warped = realNVP.forward(samples);

	std::cout << "Step 4 : Reading the results" << std::endl;
	for (auto& sample : warped) {
		std::cout << sample.first << ", " << sample.second << std::endl;
	}

	std::system("pause");
	return 0;
}

/*
int main_old()
{
	// Démarrage de l'interpréteur Python
	py::scoped_interpreter guard{};

	// Lecture du fichier Python
	//std::ifstream ifs("../../../pysrc/tutoRealNVP.py");
	//std::string pycode((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));

	py::object scope = py::module_::import("__main__").attr("__dict__");
	try {
		py::eval_file("../../../pysrc/tutoRealNVP.py", scope);
	}
	catch (py::error_already_set& err) {
		std::cerr << "Exception Python levée :" << std::endl;
		std::cerr << err.what() << std::endl;
	}
	for (auto& item : scope["warped_samples"])
		for (auto& item2 : item)
			std::cout << "res = " << item2.cast<float>() << std::endl;

	return 0;
}
*/
