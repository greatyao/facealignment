#include <iostream>
#include <stdio.h>
#include <string.h>

int train_main(int argc, char *argv[]);
int test_main(int argc, char *argv[]);
int live_main(int argc, char *argv[]);
int camera_main(int argc, char *argv[]);


int main(int argc, char *argv[])
{
	if (argc < 2) 
	{
		std::cout << "esr train/test/live/camera" << std::endl;
		return -1;
	}


	if(strcmp(argv[1], "train") == 0)
		return train_main(argc-2, argv+2);
	else if(strcmp(argv[1], "test") == 0)
		return test_main(argc-2, argv+2);
	else if(strcmp(argv[1], "live") == 0)
		return live_main(argc-2, argv+2);
	else if(strcmp(argv[1], "camera") == 0)
		return camera_main(argc-2, argv+2);
	else
		std::cout << "Unsupport command " << argv[2] << std::endl;

	return 0;
}


