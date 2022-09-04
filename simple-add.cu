#include <cstdio>
#include <cstdlib>  // std::exit

void check(const cudaError_t err, const char *msg = "unknown") {
    if (err == cudaSuccess) {
        std::printf("succeeded %s\n", msg);
    } else {
        std::fprintf(stderr, "failed %s with err = %s\n",
                     msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

template <typename T> class DevPtr {  // "device pointer"
    T *ptr;
    public:
    DevPtr() : ptr(0) {
        check(cudaMalloc(&ptr, sizeof *ptr), "construct");
    }
    T *get() const {
        return ptr;
    }
    const T value() const {
        T val;
        check(cudaMemcpy(&val, get(), sizeof val, cudaMemcpyDeviceToHost), "value");
        return val;
    }
    ~DevPtr() {
        check(cudaFree(ptr), "destruct");
    }
};

__global__ void add(int *c, const int a, const int b) {
    *c = a + b;
}

int main() {
    DevPtr<int> dp;
    add<<<1,1>>>(dp.get(), 5, 7);
    std::printf("foo %d\n", dp.value());
}
