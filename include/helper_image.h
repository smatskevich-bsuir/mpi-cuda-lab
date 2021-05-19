/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 // These are helper functions for the SDK samples (image,bitmap)
#ifndef COMMON_HELPER_IMAGE_H_
#define COMMON_HELPER_IMAGE_H_

#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const unsigned int PGMHeaderSize = 0x40;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#endif

inline bool __loadPPM(const char *file, unsigned char **data, unsigned int *w,
	unsigned int *h, unsigned int *channels) {
	FILE *fp = NULL;

	if (FOPEN_FAIL(FOPEN(fp, file, "rb"))) {
		std::cerr << "__LoadPPM() : Failed to open file: " << file << std::endl;
		return false;
	}

	// check header
	char header[PGMHeaderSize];

	if (fgets(header, PGMHeaderSize, fp) == NULL) {
		std::cerr << "__LoadPPM() : reading PGM header returned NULL" << std::endl;
		return false;
	}

	if (strncmp(header, "P5", 2) == 0) {
		*channels = 1;
	}
	else if (strncmp(header, "P6", 2) == 0) {
		*channels = 3;
	}
	else {
		std::cerr << "__LoadPPM() : File is not a PPM or PGM image" << std::endl;
		*channels = 0;
		return false;
	}

	// parse header, read maxval, width and height
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int maxval = 0;
	unsigned int i = 0;

	while (i < 3) {
		if (fgets(header, PGMHeaderSize, fp) == NULL) {
			std::cerr << "__LoadPPM() : reading PGM header returned NULL"
				<< std::endl;
			return false;
		}

		if (header[0] == '#') {
			continue;
		}

		if (i == 0) {
			i += SSCANF(header, "%u %u %u", &width, &height, &maxval);
		}
		else if (i == 1) {
			i += SSCANF(header, "%u %u", &height, &maxval);
		}
		else if (i == 2) {
			i += SSCANF(header, "%u", &maxval);
		}
	}

	// check if given handle for the data is initialized
	if (NULL != *data) {
		if (*w != width || *h != height) {
			std::cerr << "__LoadPPM() : Invalid image dimensions." << std::endl;
		}
	}
	else {
		cudaMallocHost(data, sizeof(unsigned char) * width * height *
			*channels);
		*w = width;
		*h = height;
	}

	// read and close file
	if (fread(*data, sizeof(unsigned char), width * height * *channels, fp) ==
		0) {
		std::cerr << "__LoadPPM() read data returned error." << std::endl;
	}

	fclose(fp);

	return true;
}

bool __savePPM(const char *file, unsigned char *data, unsigned int w,
	unsigned int h, unsigned int channels) {
	assert(NULL != data);
	assert(w > 0);
	assert(h > 0);

	std::fstream fh(file, std::fstream::out | std::fstream::binary);

	if (fh.bad()) {
		std::cerr << "__savePPM() : Opening file failed." << std::endl;
		return false;
	}

	if (channels == 1) {
		fh << "P5\n";
	}
	else if (channels == 3) {
		fh << "P6\n";
	}
	else {
		std::cerr << "__savePPM() : Invalid number of channels." << std::endl;
		return false;
	}

	fh << w << "\n" << h << "\n" << 0xff << std::endl;

	for (unsigned int i = 0; (i < (w * h * channels)) && fh.good(); ++i) {
		fh << data[i];
	}

	fh.flush();

	if (fh.bad()) {
		std::cerr << "__savePPM() : Writing data failed." << std::endl;
		return false;
	}

	fh.close();

	return true;
}

#endif  // COMMON_HELPER_IMAGE_H_#pragma once
