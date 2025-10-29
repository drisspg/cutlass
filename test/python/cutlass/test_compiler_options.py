#################################################################################################
#
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Unit tests for compiler options functionality
"""

import unittest
from cutlass.backend.compiler import ArtifactManager


class TestCompilerOptions(unittest.TestCase):
    """
    Test suite for compiler options functionality
    """

    def test_initialization(self):
        """Test that ArtifactManager initializes with empty additional options"""
        compiler = ArtifactManager()
        self.assertEqual(compiler.get_additional_options(), [])

    def test_set_additional_options(self):
        """Test setting additional compiler options"""
        compiler = ArtifactManager()
        compiler.set_additional_options(["-lineinfo", "-G"])
        self.assertEqual(compiler.get_additional_options(), ["-lineinfo", "-G"])

    def test_set_additional_options_single_string(self):
        """Test setting a single additional compiler option as a string"""
        compiler = ArtifactManager()
        compiler.set_additional_options("-lineinfo")
        self.assertEqual(compiler.get_additional_options(), ["-lineinfo"])

    def test_add_additional_options(self):
        """Test adding additional compiler options"""
        compiler = ArtifactManager()
        compiler.add_additional_options("-lineinfo")
        self.assertEqual(compiler.get_additional_options(), ["-lineinfo"])
        compiler.add_additional_options(["-G", "-O3"])
        self.assertEqual(compiler.get_additional_options(), ["-lineinfo", "-G", "-O3"])

    def test_enable_debug_info(self):
        """Test enabling debug info"""
        compiler = ArtifactManager()
        compiler.enable_debug_info()
        self.assertIn("-lineinfo", compiler.get_additional_options())

    def test_enable_debug_info_idempotent(self):
        """Test that enabling debug info multiple times doesn't add duplicates"""
        compiler = ArtifactManager()
        compiler.enable_debug_info()
        compiler.enable_debug_info()
        options = compiler.get_additional_options()
        self.assertEqual(options.count("-lineinfo"), 1)

    def test_disable_debug_info(self):
        """Test disabling debug info"""
        compiler = ArtifactManager()
        compiler.enable_debug_info()
        self.assertIn("-lineinfo", compiler.get_additional_options())
        compiler.disable_debug_info()
        self.assertNotIn("-lineinfo", compiler.get_additional_options())

    def test_disable_debug_info_removes_all_occurrences(self):
        """Test that disable_debug_info removes all -lineinfo flags"""
        compiler = ArtifactManager()
        compiler.add_additional_options(["-lineinfo", "-G", "-lineinfo"])
        compiler.disable_debug_info()
        self.assertNotIn("-lineinfo", compiler.get_additional_options())
        self.assertIn("-G", compiler.get_additional_options())

    def test_options_preserved_across_backend_switch(self):
        """Test that additional options are preserved when switching backends"""
        compiler = ArtifactManager()
        compiler.enable_debug_info()
        compiler.nvrtc()
        self.assertIn("-lineinfo", compiler.get_additional_options())
        compiler.nvcc()
        self.assertIn("-lineinfo", compiler.get_additional_options())

    def test_get_additional_options_returns_copy(self):
        """Test that get_additional_options returns a copy of the list"""
        compiler = ArtifactManager()
        compiler.add_additional_options("-lineinfo")
        options = compiler.get_additional_options()
        options.append("-G")
        # Original should not be modified
        self.assertEqual(compiler.get_additional_options(), ["-lineinfo"])


if __name__ == '__main__':
    unittest.main()
