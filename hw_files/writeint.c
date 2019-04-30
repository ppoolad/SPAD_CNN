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

int main(int argc, char **argv) {
	if (argc != 3) {
		puts("Usage: (1) writeint 0xADDR VALUE: Writes (base 10) VALUE at (base 16) ADDR on \n"
			 "                                  AXIlite bus connecting to FPGA design\n\n"
			 "       (2) writeint file FILEPATH: uses file at FILEPATH with following format:\n"
			 "                                   0xADDR1 VALUE1\n"
			 "                                   0xADDR2 VALUE2\n"
			 "                                   ...\n"
			 "                                   0xADDRn VALUEn");
		return 1;
	}
	
	
	//Forward declare variables
	FILE *fp = NULL;
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
		close(ctrl_reg_fd);
		return 0;
	}
	
	//mmap the device file
	map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_reg_fd, 0);

	if (map_base == (void *)-1) {
		perror("Could not mmap device file");
		munmap(map_base, MAP_SIZE);
		return 0;
	}
	
	if (!strcmp(argv[1], "file")) {
		fp = fopen(argv[2], "rb");
		if (fp == NULL) {
			char line[256];
			sprintf(line, "Could not open file %s", argv[2]);
			perror(line);
			fclose(fp);
			return 0;
		}
		
		char line[80];
		//read params file and pass the values to writeint
		while (fgets(line, 80, fp) != NULL) {
			if (line[0] == '\n') continue; //Skip empty lines
			int addr = -1, value = -1;
			sscanf(line, "%x %d", &addr, &value);
			if (addr < 0 || value < 0) {
				printf("Error: invalid format \"%s\"", line);
				return 0;
			}
			writeint(map_base, addr, value);
			//msync(map_base, MAP_SIZE, MS_SYNC); //Wait until value is actually read
			printf("Wrote %d to address 0x%x\n", value, addr);
		}
	} else {
		int addr = -1, value = -1;
		sscanf(argv[1], "%x", &addr);
		if (addr < 0) {
			printf("Error: invalid address \"%s\"", argv[1]);
			return 0;
		}
		sscanf(argv[2], "%d", &value);
		if (value < 0) {
			printf("Error: invalid value \"%s\"", argv[2]);
			return 0;
		}
		writeint(map_base, addr, value);
		printf("Wrote %d to address 0x%x\n", value, addr);
	}
	
	return 0;
}

