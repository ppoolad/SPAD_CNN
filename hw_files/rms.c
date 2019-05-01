#include <stdio.h>

int main(int argc, char **argv) {
        FILE *fp1, *fp2;
        fp1 = fopen(argv[1],"rb");
        if (fp1 == NULL) {
                char line[80];
                sprintf(line, "Could not read file %s", argv[1]);
                perror(line);
                return -1;
        }
        fp2 = fopen(argv[2],"rb");
        if (fp2 == NULL) {
                char line[80];
                sprintf(line, "Could not read file %s", argv[2]);
                perror(line);
                return -1;
        }
        float total = 0.0f;
        int totalOutputs = 0;
        int badOutputs = 0;
        while(!feof(fp1) && !feof(fp2)) {
                float f1, f2;
                fread(&f1, sizeof(float), 1, fp1);
                fread(&f2, sizeof(float), 1, fp2);
                float diff = f1 - f2;


                diff *= diff;
                total += diff;
                totalOutputs++;

                if (diff > 1e-2) badOutputs++;
        }
        printf("Mean squared error: %g\n(%d bad outputs out of %d total outputs)\n", total/totalOutputs, badOutputs, totalOutputs);
        fclose(fp1);
        fclose(fp2);
        return 0;
}
