#include <iostream>
#include <random>

int main()
{
    std::mt19937 gen(42);
    uint32_t x = gen(), a = gen();
    uint64_t z = static_cast<uint64_t>(x) * static_cast<uint64_t>(a);

    uint32_t xl = x & 0xffff, al = a & 0xffff;
    x = x >> 16, a = a >> 16;

    uint32_t m0 = static_cast<int32_t>(xl) * static_cast<int32_t>(al);
    uint32_t m1 = static_cast<int32_t>(xl) * static_cast<int32_t>(a);
    uint32_t m2 = static_cast<int32_t>(x) * static_cast<int32_t>(al);
    uint32_t m3 = static_cast<int32_t>(x) * static_cast<int32_t>(a);

    uint32_t m1l = m1 & 0xffff, m2l = m2 & 0xffff, m0l = m0 & 0xffff;
    m1 = m1 >> 16, m2 = m2 >> 16, m0 = m0 >> 16;

    uint32_t mm = static_cast<int32_t>(m1l) + static_cast<int32_t>(m2l) + static_cast<int32_t>(m0);
    uint32_t mml = mm << 16;
    mm = mm >> 16;

    uint32_t l = mml | m0l;
    uint32_t h = static_cast<int32_t>(m3) + static_cast<int32_t>(m2) + static_cast<int32_t>(m1) + static_cast<int32_t>(mm);

    uint64_t y = (static_cast<uint64_t>(h) << 32) | l;

    std::cout << "x: " << x << ", a: " << a << std::endl;
    std::cout << "y: " << y << ", z: " << z << std::endl;
    return 0;
}