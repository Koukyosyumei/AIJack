#include <pybind11/pybind11.h>

namespace scipy11
{
    namespace scipy
    {

        class special_module : public pybind11::module
        {
        public:
            using pybind11::module::module;

            template <class... TArgs>
            pybind11::object binom(TArgs &&...args)
            {
                return attr("binom")(std::forward<TArgs>(args)...);
            }
        };

        special_module import_special()
        {
            return pybind11::module::import("scipy.special");
        }
    }
}
