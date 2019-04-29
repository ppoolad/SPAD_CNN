#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <byteswap.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>
#include <termios.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/unistd.h>

#define MAP_SIZE 1024UL*1024UL

/* ltoh: little to host */
/* htol: little to host */
#if __BYTE_ORDER == __LITTLE_ENDIAN
#  define ltohl(x)       (x)
#  define ltohs(x)       (x)
#  define htoll(x)       (x)
#  define htols(x)       (x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#  define ltohl(x)     __bswap_32(x)
#  define ltohs(x)     __bswap_16(x)
#  define htoll(x)     __bswap_32(x)
#  define htols(x)     __bswap_16(x)
#endif

static void writeint(volatile void* map_base, int offset, int value)
{
  volatile void* virt_addr = (volatile void*)((char*)map_base + offset);
  *((uint32_t *) virt_addr) = htoll(value);
}
static int readint(volatile void* map_base, int offset)
{
  volatile void* virt_addr = (volatile void*)((char*)map_base + offset);
  return ltohl(*((uint32_t *) virt_addr));
}

volatile int stop;
void catchSIGINT(int signum) {
	stop = 1;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		puts("Usage: runip 0xADDR: Using standard IP block protocol, start IP core whose \n"
			 "                     control register is at (base 16) ADDR, and wait for it to finish\n");
		return 1;
	}
	
	signal(SIGINT, catchSIGINT); //Set up custom signal handler
	
	int ret = 0;
	
	//Forward declare variables
	void *map_base = NULL;
	
	//Get correct prefix for device driver files
	char const *xdma_base = getenv("XDMA");
	
	char ctrl_reg_path[80];
	sprintf(ctrl_reg_path, "%s_user", xdma_base);
	int ctrl_reg_fd = open(ctrl_reg_path, O_RDWR | O_SYNC);
	if (ctrl_reg_fd == -1) {
		char line[80];
		sprintf(line, "Could not open file %s", ctrl_reg_path);
		perror(line);
		goto cleanup;
	}
	
	//mmap the device file
	map_base = mmap(
		NULL, //No address hint
		MAP_SIZE, //1 Mb mapping. Will this work?
		PROT_READ | PROT_WRITE, //Allow reading and writing
		MAP_SHARED, //Allows multiple programs to access the resource. Not really important for us
		ctrl_reg_fd, //File descriptor
		0 //Offset into device file
	);
	if (map_base == (void *)-1) {
		perror("Could not mmap device file");
		goto cleanup;
	}
	
	int addr = -1;
	sscanf(argv[1], "%x", &addr);
	if (addr == -1) {
		printf("Error: invalid address\"%s\"\n", argv[1]);
		goto cleanup;
	}
	
	int timeLeft = 2000;
	if(!(readint(map_base, addr) & 0xe)) {
		puts("Warning: device has not signalled that it is ready.\nStarting anyway in 10 seconds (CTRL-C to abort)");
		do {
			if (timeLeft % 20 == 0) {
				printf("\rTime left: %.1f seconds", 0.005f*timeLeft);
				fflush(stdout);
			}
			usleep(5000); //Sleep for 5 ms
			timeLeft--; //Simple way to gauge how much time the computation took
		} while (!(readint(map_base, addr) & 0xe) && !stop && timeLeft != 0);
		if (stop) {
			puts("Aborting...");
			goto cleanup;
		} else if (readint(map_base, addr) & 0xe) {
			puts("Device has signalled it is ready. Continuing...");
		} else if (timeLeft == 0) {
			puts("BIG WARNING: device still not ready, but starting anyway.");
		}
	}

	printf("Starting ip at address 0x%x...\n", addr);
	writeint(map_base, addr, 0x1); //Write 1 to start bit
	int sleepcounts = 0;
	do {
		if (sleepcounts % 20 == 0) {
			printf("\rTime waited: %.1f seconds", 0.005f*sleepcounts);
			fflush(stdout);
		}
		usleep(5000); //Sleep for 5 ms
		sleepcounts++; //Simple way to gauge how much time the computation took
	} while (!(readint(map_base, addr) & 0xe) && !stop);
	
	if (stop == 1) {
		puts("\nInterrupted! Did not wait for IP core to finish");
		goto cleanup;
	}
	printf("\nApproximate elapsed time: %g seconds\n", 0.005f*sleepcounts);
	
	cleanup:`
	if (map_base != NULL) munmap(map_base, MAP_SIZE);
	if (ctrl_reg_fd != -1) close(ctrl_reg_fd);
	return ret;
}
