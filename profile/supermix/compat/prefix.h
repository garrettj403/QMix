// Forced-include header for building SuperMix (written for g++ 2.x-5) with
// modern clang/libc++. libc++ pulls <complex> in transitively, so SuperMix's
// "typedef Complex complex;" clashes with std::complex under
// "using namespace std;". Load every std header SuperMix uses first, then
// rename SuperMix's "complex" alias to its underlying "Complex" class.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#define complex Complex
#define ratio supermix_ratio  // std::ratio clash in supcond.cc
