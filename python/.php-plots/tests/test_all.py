# coding: utf-8

__all__ = ["ExecutablesTest"]

# adjust the path to import the executables
import os
import sys
test_dir = os.path.dirname(os.path.abspath(__file__))
bin_dir = os.path.join(os.path.dirname(test_dir), "bin")
sys.path.insert(0, bin_dir)

import unittest
import shutil
import tempfile


class ExecutablesTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.example_pdf = os.path.join(test_dir, "files", "example.pdf")

    def touch(self, path):
        # the file must not exist
        if os.path.exists(path):
            raise RuntimeError(f"file {path} already exists")

        # create the directory if it does not exist
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # write the file
        with open(path, "w") as f:
            f.write("")

        return path

    def check_existing(self, *path):
        return self.assertTrue(os.path.exists(os.path.join(*map(str, path))))

    def check_missing(self, *path):
        return self.assertFalse(os.path.exists(os.path.join(*map(str, path))))

    def test_copy_index(self):
        from pb_copy_index import copy_index

        # shallow copy
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(os.path.join(tmp_dir, "l1", "l2"))
            copy_index([tmp_dir, os.path.join(tmp_dir, "l1", "l2")], recursive=False)
            self.check_existing(tmp_dir, "index.php")
            self.check_missing(tmp_dir, "l1", "index.php")
            self.check_existing(tmp_dir, "l1", "l2", "index.php")

        # recursive copy
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(os.path.join(tmp_dir, "l1", "l2"))
            copy_index([tmp_dir, os.path.join(tmp_dir, "l1", "l2")], recursive=True)
            self.check_existing(tmp_dir, "index.php")
            self.check_existing(tmp_dir, "l1", "index.php")
            self.check_existing(tmp_dir, "l1", "l2", "index.php")

        # expect failure
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self.touch(os.path.join(tmp_dir, "file.txt"))

            with self.assertRaises(RuntimeError):
                copy_index([path])

    def test_pdf_to_png(self):
        from pb_pdf_to_png import pdf_to_png

        # shallow lookup
        with tempfile.TemporaryDirectory() as tmp_dir:
            # copy the example files into deep subdirectories
            os.makedirs(os.path.join(tmp_dir, "l1", "l2"))
            dirs = [tmp_dir, os.path.join(tmp_dir, "l1", "l2"), os.path.join(tmp_dir, "l1")]
            for d in dirs:
                shutil.copy2(self.example_pdf, d)
            # convert only some pdfs, single core
            pdf_to_png(dirs[:2], recursive=False, n_cores=1)
            # check pdfs
            self.check_existing(tmp_dir, "example.png")
            self.check_missing(tmp_dir, "l1", "example.png")
            self.check_existing(tmp_dir, "l1", "l2", "example.png")

        # recursive lookup
        with tempfile.TemporaryDirectory() as tmp_dir:
            # copy the example files into deep subdirectories
            os.makedirs(os.path.join(tmp_dir, "l1", "l2"))
            dirs = [tmp_dir, os.path.join(tmp_dir, "l1", "l2"), os.path.join(tmp_dir, "l1")]
            for d in dirs:
                shutil.copy2(self.example_pdf, d)
            # convert recursively, two cores
            pdf_to_png(tmp_dir, recursive=True, n_cores=2)
            # check pdfs
            self.check_existing(tmp_dir, "example.png")
            self.check_existing(tmp_dir, "l1", "example.png")
            self.check_existing(tmp_dir, "l1", "l2", "example.png")

    def test_deploy_plots(self):
        from pb_deploy_plots import deploy_plots

        with tempfile.TemporaryDirectory() as tmp_src_dir:
            # copy the example files into deep subdirectories
            os.makedirs(os.path.join(tmp_src_dir, "l1", "l2"))
            dirs = [tmp_src_dir, os.path.join(tmp_src_dir, "l1", "l2")]
            for d in dirs:
                shutil.copy2(self.example_pdf, d)
            # create additional files
            self.touch(os.path.join(tmp_src_dir, "l1", "file.txt"))
            self.touch(os.path.join(tmp_src_dir, "l1", "file.unknown"))

            # deploy shallow
            with tempfile.TemporaryDirectory() as tmp_dst_dir:
                # deploy and check files
                deploy_plots(dirs, tmp_dst_dir, recursive=False)
                src_base = os.path.basename(tmp_src_dir)
                self.check_existing(tmp_dst_dir, "index.php")
                self.check_existing(tmp_dst_dir, src_base, "index.php")
                self.check_existing(tmp_dst_dir, src_base, "example.pdf")
                self.check_existing(tmp_dst_dir, "l2", "index.php")
                self.check_existing(tmp_dst_dir, "l2", "example.pdf")
                self.check_missing(tmp_dst_dir, "l1")

            # deploy recursive
            with tempfile.TemporaryDirectory() as tmp_dst_dir:
                # deploy and check files
                deploy_plots(os.path.join(tmp_src_dir, "*"), tmp_dst_dir, recursive=True)
                self.check_existing(tmp_dst_dir, "index.php")
                self.check_existing(tmp_dst_dir, "example.pdf")
                self.check_existing(tmp_dst_dir, "l1", "index.php")
                self.check_existing(tmp_dst_dir, "l1", "file.txt")
                self.check_existing(tmp_dst_dir, "l1", "file.unknown")
                self.check_existing(tmp_dst_dir, "l1", "l2", "index.php")
                self.check_existing(tmp_dst_dir, "l1", "l2", "example.pdf")

            # deploy recursive, filter extensions
            with tempfile.TemporaryDirectory() as tmp_dst_dir:
                # deploy and check files
                deploy_plots(
                    os.path.join(tmp_src_dir, "*"),
                    tmp_dst_dir,
                    recursive=True,
                    extensions=("txt", "pdf"),
                )
                self.check_existing(tmp_dst_dir, "index.php")
                self.check_existing(tmp_dst_dir, "example.pdf")
                self.check_existing(tmp_dst_dir, "l1", "index.php")
                self.check_existing(tmp_dst_dir, "l1", "file.txt")
                self.check_missing(tmp_dst_dir, "l1", "file.unknown")
                self.check_existing(tmp_dst_dir, "l1", "l2", "index.php")
                self.check_existing(tmp_dst_dir, "l1", "l2", "example.pdf")
