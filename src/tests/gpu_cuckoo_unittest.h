/*
 * gpu_cuckoo_unittest.h
 *
 *  Created on: 01-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef GPU_CUCKOO_UNITTEST_H_
#define GPU_CUCKOO_UNITTEST_H_

#include <gtest/gtest.h>

class GpuCuckooTest: public testing::Test
{
protected:
	GpuCuckooTest(){}
    ~GpuCuckooTest(){}

    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
    }
};

#endif /* GPU_CUCKOO_UNITTEST_H_ */
