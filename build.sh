#bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
#
#echo "----------------- build OK ----------------"

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
echo "----------------- build pip OK ----------------"

pip install /tmp/tensorflow_pkg/tensorflow-1.1.0rc0-cp27-cp27mu-linux_x86_64.whl
echo "----------------- install pip OK ----------------"
