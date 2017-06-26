cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

set(source src/ESRMain.cpp 
	src/ESRTest.cpp 
	src/ESRTrain.cpp 
	src/ESRRegressor.cpp 
	src/ESRUtils.cpp 
	src/ESRCascadedFerns.cpp 
	src/ESRFern.cpp 
	src/ESRFaceDetector.cpp)
add_executable(esr ${source})
target_link_libraries(esr ${OpenCV_LIBS})

