## Руководство по использованию (инструкция по инсталляции)

Настоящее руководство составлено для установки и запуска программного обеспечения компании GuaranaCam (ПО) для подсчета посетителей.
Для корректной работы ПО необходим компьютер со следующими минимальными характеристиками:

- Ubuntu 16.04.4 LTS x64;
- GPU компании Nvidia с compute capability не ниже 6.0.

Для установки ПО на компьютер с вновь установленной «пустой» ОС Ubuntu 16.04.4 LTS требуется выполнить следующие действия:

### 1)	Установить весь стек технологий NVIDIA для поддержки Deep Learning:

- Nvidia CUDA driver;
- Nvidia CUDA Toolkit версии не ниже cuda-9.0;
- Библиотека cuDNN версии не ниже 7.0;

Для установки вышеуказанных модулей необходимо следовать инструкции на [сайте](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) компании Nvidia:

    https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

### 2)	Установить библиотеки для Python 3:

    sudo apt install python3-pip
    pip3 install numpy
    pip3 install pandas
    pip install opencv-python
    #pip3 install --upgrade tensorflow-gpu
    pip3 install cython
    pip3 install scikit-image
    sudo apt-get install python3-tk
    pip3 install flask
    pip3 install sklearn


### 3)	Установка зависимостей:

- libcupti

    sudo apt-get install libcupti-dev
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

- Связанные с Python 3.X

    sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel

### 4)	Настройка Tensorflow для сборки из исходного кода:

Скачиваем bazel

    cd ~/
    wget https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-linux-x86_64.sh
    chmod +x bazel-0.14.0-installer-linux-x86_64.sh
    ./bazel-0.14.0-installer-linux-x86_64.sh --user
    echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
    
Перезагружаем environment variables
    
    source ~/.bashrc
    sudo ldconfig
    
Начинаем процесс установки TensorFlow, скачивая последнюю версию tensorflow 1.8.0 (на момент написания инструкции).

    cd ~/
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    git pull
    git checkout r1.8
    ./configure

Укажите путь к Python3

    Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3

Нажмите enter два раза, затем отвечайте на вопросы конфигуратора. Вопросы (и ваши ответы на них) могут немного отличаться.

    Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: Y
    Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: Y
    Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: Y
    Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: Y
    Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: Y
    Do you wish to build TensorFlow with XLA JIT support? [y/N]: N
    Do you wish to build TensorFlow with GDR support? [y/N]: N
    Do you wish to build TensorFlow with VERBS support? [y/N]: N
    Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
    Do you wish to build TensorFlow with CUDA support? [y/N]: Y
    Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.2
    Please specify the location where CUDA 9.2 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda-9.2
    Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.1.4
    Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-9.2]: /usr/local/cuda-9.2
    Do you wish to build TensorFlow with TensorRT support? [y/N]: N
    Please specify the NCCL version you want to use. [Leave empty to default to NCCL 1.3]: 2.2
    Please specify the location where NCCL 2 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-9.2]: /usr/local/cuda-9.2/targets/x86_64-linux
    
Теперь нужно указать Copmute Capability вашего GPU:

    Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 6.0] 6.0
    Do you want to use clang as CUDA compiler? [y/N]: N
    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: /usr/bin/gcc
    Do you wish to build TensorFlow with MPI support? [y/N]: N
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -march=native
    Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:N

На этом конфигурация установки закончена.

### 5)	Сборка Tensorflow с использованием bazel:

Следующий шаг в уставке Tensorflow с поддержкой GPU– это сборка tensorflow при помощи мощной системы сборки bazel.
Чтобы собрать pip-пакет для TensorFlow, необходимо выполнить следующую команду:
    
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    
Этот процесс может длиться достаточно продолжительное время (от 40 минут).
Также, если у вас будет ошибка Segmentation Fault, попробуйте запустить команду выше ещё раз, обычно помогает.
Bazel собирает скрипт с названием build_pip_package. Запуск этого скрипта приведет к сборке файла *.whl внутри директории tensorflow_pkg:
Чтобы построить whl-файл:

    bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg

Устанавливаем tensorflow при помощи pip:

    cd tensorflow_pkg
    
~~для существующего виртуального окружения…:'~~
    
    pip3 install tensorflow*.whl
    
~~или для нового виртуального окружения с использованием virtualenv:~~
    
    sudo apt-get install virtualenv
    virtualenv tf_1.8.0_cuda9.2 -p /usr/bin/python3
    source tf_1.8.0_cuda9.2/bin/activate
    pip install tensorflow*.whl
    
**для python 3: (используйте sudo, если потребуется)**
    
    pip3 install tensorflow*.whl
    
Если возникнет ошибка unsupported platform, тогда убедитесь, что вы запускаете верную версию pip, ассоциированную с python, который был использован при конфигурации сборки tensorflow.

### 6)	Проверка установки Tensorflow с поддержкой GPU: 

Запускайте в терминале:
    
    python3
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

Если в выводе появится следующее сообщение, то всё установлено верно.
    
    Hello, TensorFlow!
