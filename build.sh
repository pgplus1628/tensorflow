#bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
#
#echo "----------------- build OK ----------------"

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
echo "----------------- build pip OK ----------------"

#pip uninstall tensorflow
#pip uninstall tensorflow
#
#pip install /tmp/tensorflow_pkg/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl
#echo "----------------- install pip OK ----------------"
