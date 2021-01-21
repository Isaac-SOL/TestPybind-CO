// TestPybind : Test de fonctionnement de l'embed Python dans C++
// Par Pierre Cuquel

#include "TestPybind.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

int main()
{
	std::cout << "Hello from C++" << std::endl;
	py::scoped_interpreter guard{};

	auto locals = py::dict("res"_a = "test");
	py::exec(R"pydelim(
		import numpy as np
		res = np.array([[1, 2, 3], [4, 5, 6]])
	)pydelim", py::globals(), locals);
	for (auto& item : locals["res"])
		for (auto& item2 : item)
			std::cout << "res = " << item2.cast<int>() << std::endl;

	return 0;
}
