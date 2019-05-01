#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void read_file(char const *xdma_base, char const* filepath, int addr, int len) {
//Forward declare variable to avoid goto error
	FILE *fp = NULL;
	int dma_from_device_fd;
	char *data = NULL;
	
	//concatenate "_c2h_0" to get char device path
	char path[80];
	sprintf(path, "%s_c2h_0", xdma_base);
	
	//Open char device file
	dma_from_device_fd = open(path, O_RDWR | O_NONBLOCK);
	
	//Check if device file opened succesfully
	if (dma_from_device_fd == -1) {
		char line[80];
		sprintf(line, "Could not open file %s", path);
		perror(line);
		close(dma_from_device_fd);
		return;
	}
	
	//Open selected file
	fp = fopen(filepath, "wb");
	if (!fp) {
		char line[80];
		sprintf(line, "Could not open file %s", filepath);
		perror(line);
		fclose(fp);
		return;
	}
	
	//aligned memory allocation
	posix_memalign((void**) &data, 4096, len + 4096); 
	
	//Read data from FPGA memory over PCI 
	lseek(dma_from_device_fd, addr, SEEK_SET);
	read(dma_from_device_fd, data, len);
	
	//write the read data into a file
	fwrite(data, 1, len, fp);

	printf("Read %d bytes from address %d (0x%x in hex)\n", len, addr, addr);
}


int main(int argc, char **argv) {
	int addr, len;
	char const *xdma_base = getenv("XDMA"); //Is /dev/xdma1 on my machine
	sscanf(argv[2], "%d", &addr);
	sscanf(argv[3], "%d", &len);
	read_file(xdma_base, argv[1], addr, len);
	return 0;
}
