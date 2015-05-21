/*
 * constants.cuh
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef CONSTANTS_CUH_
#define CONSTANTS_CUH_

#include <stdlib.h>

template<unsigned N>
struct Constants
{
	unsigned values[N];
	unsigned const size = N;

	void init(unsigned constants[N]){ values = constants; }
	void initRandom(){ for(int i=0; i<N; i++) values[i] = rand(); }
};

#endif /* CONSTANTS_CUH_ */
