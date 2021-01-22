// TestPybind : Test de fonctionnement de l'embed Python dans C++
// Par Pierre Cuquel

#include "TestPybind.h"
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

int main()
{
	// Démarrage de l'interpréteur Python
	py::scoped_interpreter guard{};

	// Lecture du fichier Python
	std::ifstream ifs("../../../pysrc/tutoRealNVP.py");
	std::string pycode((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));

	auto locals = py::dict("res"_a = "test");
	try {
		py::exec(pycode, py::globals(), locals);
	}
	catch (py::error_already_set& err) {
		std::cerr << "Exception Python levée :" << std::endl;
		std::cerr << err.what() << std::endl;
	}
	for (auto& item : locals["warped_samples"])
		for (auto& item2 : item)
			std::cout << "res = " << item2.cast<int>() << std::endl;

	return 0;
}
