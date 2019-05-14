//send data through PCIe to DDR memory

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


void write_file (char const *xdma_base, char const* filepath, int addr) {
	FILE *fp = NULL;
	int dma_to_device_fd;
	char *data = NULL;
	int len;

	char path[80];
	sprintf(path, "%s_h2c_0", xdma_base);
	dma_to_device_fd = open(path, O_RDWR | O_NONBLOCK);
	printf("Openning %s\n", path);
	//Check if device file opened succesfully
	if (dma_to_device_fd == -1) {
		char line[80];
		sprintf(line, "Could not open file %s", path);
		perror(line);
		close(dma_to_device_fd);
		dma_to_device_fd = -1;
		return;
	}

	//Open file with read access
	fp = fopen(filepath, "rb");
	printf("Openning %s\n", filepath);
	if (!fp) {
		char line[80];
		sprintf(line, "Could not open file %s", filepath);
		perror(line);
		fclose(fp);
		return;
	}
	
	//find file length
	fseek(fp, 0, SEEK_END);
	len = ftell(fp);
	rewind(fp);

	//aligned memory allocation
	posix_memalign((void**) &data, 4096, len + 4096); 
	fread(data, 1, len, fp);

	//Send data over PCI to FPGA
	printf("Sending to PCIe \n\r");
	lseek(dma_to_device_fd, addr, SEEK_SET);
	printf("Sending to PCIe.. \n\r");
	write(dma_to_device_fd, data, len);

	printf("Wrote %d bytes to address %d (0x%x in hex)\n", len, addr, addr);
}


int main(int argc, char **argv) {
	int addr;
	char const *xdma_base = getenv("XDMA"); //Is /dev/xdma1 on my machine
	sscanf(argv[2], "%d", &addr);
	write_file(xdma_base, argv[1], addr);
	return 0;
}