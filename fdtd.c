#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "fdtd.h"

int world_size; 
int rank, cart_rank; 

int dims[3]    = {0, 0, 0};
int periods[3] = {0, 0, 0};

int reorder = 0;

int coords[3];
int neighbors[6];

MPI_Comm cart_comm;

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    printf("\nUsage: ./fdtd <param_file>\n\n");
    exit(1);
  }

  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Dims_create(world_size, 3, dims);

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);
  MPI_Comm_rank(cart_comm, &cart_rank);

  MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

  MPI_Cart_shift(cart_comm, 0, 1, &neighbors[UP], &neighbors[DOWN]);

  MPI_Cart_shift(cart_comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
  
  MPI_Cart_shift(cart_comm, 2, 1, &neighbors[IN], &neighbors[OUT]);

  printf("Rank = %4d - Coords = (%3d, %3d, %3d)"
        " - Neighbors (up, down, left, right, in, out) = (%3d, %3d, %3d, %3d, %3d, %3d)\n",
            rank, coords[0], coords[1],coords[2], 
            neighbors[UP], neighbors[DOWN], neighbors[LEFT], neighbors[RIGHT], neighbors[IN], neighbors[OUT]);

  simulation_data_rank_t simdata_rank;
  init_simulation(&simdata_rank, argv[1]);
  

  int numtimesteps = floor(simdata_rank.params.maxt / simdata_rank.params.dt);

  double start = GET_TIME();
  
  for (int tstep = 0; tstep <= numtimesteps; tstep++) {
    
    apply_source(&simdata_rank, tstep);
   
    if (simdata_rank.params.outrate > 0 && (tstep % simdata_rank.params.outrate) == 0) {
      
      for (int i = 0; i < simdata_rank.params.numoutputs; i++) {
        data_rank_t *output_data = NULL;
        
        switch (simdata_rank.params.outputs[i].source) {
        case PRESSURE:
          output_data = simdata_rank.pold;
          break;
        case VELOCITYX:
          output_data = simdata_rank.vxold;
          break;
        case VELOCITYY:
          output_data = simdata_rank.vyold;
          break;
        case VELOCITYZ:
          output_data = simdata_rank.vzold;
          break;
        default:
          break;
        }
        
        double time = tstep * simdata_rank.params.dt;
        
        write_output(&simdata_rank, &simdata_rank.params.outputs[i], output_data, tstep, time); 
        
      }
    }
    
    if (tstep > 0 && tstep % (numtimesteps / 10) == 0) {
      printf("step %8d/%d", tstep, numtimesteps);
      
      if (tstep != numtimesteps) {
        double elapsed_sofar = GET_TIME() - start;
        double timeperstep_sofar = elapsed_sofar / tstep;

        double eta = (numtimesteps - tstep) * timeperstep_sofar;

        printf(" (ETA: %8.3lf seconds)", eta);
      }

      printf("\n");
      fflush(stdout);
      
    }
    
    update_pressure(&simdata_rank);
    update_velocities(&simdata_rank);
    swap_timesteps(&simdata_rank);
    
  }
  
  double elapsed = GET_TIME() - start;
  double numupdates =
      (double)NUMNODESTOT(simdata_rank.pold->grid) * (numtimesteps + 1);
  double updatespers = numupdates / elapsed / 1e6;

  printf("\nElapsed %.6lf seconds (%.3lf Mupdates/s)\n\n", elapsed,
         updatespers);

  finalize_simulation(&simdata_rank);
  
  MPI_Finalize();
  return 0;
}

/******************************************************************************
 * Utilities functions                                                        *
 ******************************************************************************/

void setvalue(simulation_data_rank_t *simdata_rank, data_rank_t *data, int m_glob, int n_glob, int p_glob, double val){
  //Need to find the global numnodes
  int numnodes_globx = simdata_rank -> grid.numnodesx;
  int numnodes_globy = simdata_rank -> grid.numnodesy;
  int numnodes_globz = simdata_rank -> grid.numnodesz;

  // Check for invalid m,n,p index
  if(m_glob < 0 || n_glob < 0 || p_glob < 0 
  || m_glob > numnodes_globx 
  ||  n_glob > numnodes_globy 
  ||  p_glob > numnodes_globz){
    //DEBUG_PRINT("Error : invlaid access 1");
    }

  //defines the local m,n,p index
  int m = m_glob - simdata_rank -> grid.startm;
  int n = n_glob - simdata_rank -> grid.startn;
  int p = p_glob - simdata_rank -> grid.startp;

  //check inside the subdomain
  if(m < 0 || n < 0 || p < 0 
  || m > NUMNODESX(simdata_rank) || n > NUMNODESY(simdata_rank) || p > NUMNODESZ(simdata_rank) ) {
    //DEBUG_PRINT("Error : invlaid access 2");
    }
  
  data -> vals[INDEX3D((data)->grid, m, n, p)] = val;

}

double getvalue(simulation_data_rank_t *simdata_rank, data_rank_t *data, int m_glob, int n_glob, int p_glob){
  //Need to find the global numnodes
  int numnodes_globx = simdata_rank -> grid.numnodesx;
  int numnodes_globy = simdata_rank -> grid.numnodesy;
  int numnodes_globz = simdata_rank -> grid.numnodesz; 
  // Check for invalid m,n,p index
  if(m_glob < 0 || n_glob < 0 || p_glob < 0 
  || m_glob > numnodes_globx 
  ||  n_glob > numnodes_globy 
  ||  p_glob > numnodes_globz) return 0;

  //defines the local m,n,p index
  int m = m_glob - simdata_rank -> grid.startm;
  int n = n_glob - simdata_rank -> grid.startn;
  int p = p_glob - simdata_rank -> grid.startp;

  //check inside the subdomain
  if(m < 0 || n < 0 || p < 0 
  || m > NUMNODESX(simdata_rank) - 1|| n > NUMNODESY(simdata_rank) - 1|| p > NUMNODESZ(simdata_rank) - 1) return 0;
  
  return data -> vals[INDEX3D((data)->grid, m, n, p)];

}


char *copy_string(char *str) {
  size_t len;
  if (str == NULL || (len = strlen(str)) == 0) {
    DEBUG_PRINT("NULL of zero length string passed as argument");
    return NULL;
  }

  char *cpy;
  if ((cpy = malloc((len + 1) * sizeof(char))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return NULL;
  }

  return strcpy(cpy, str);
}

void closest_index(grid_rank_t *grid, double x, double y, double z, int *cx, int *cy,
                   int *cz) {
  int m = (int)((x - grid->xmin) / (grid->xmax - grid->xmin) * grid->numnodesx);
  int n = (int)((y - grid->ymin) / (grid->ymax - grid->ymin) * grid->numnodesy);
  int p = (int)((z - grid->zmin) / (grid->zmax - grid->zmin) * grid->numnodesz);

  *cx = (m < 0) ? 0 : (m > grid->numnodesx - 1) ? grid->numnodesx - 1 : m;
  *cy = (n < 0) ? 0 : (n > grid->numnodesy - 1) ? grid->numnodesy - 1 : n;
  *cz = (p < 0) ? 0 : (p > grid->numnodesz - 1) ? grid->numnodesz - 1 : p;
}

void print_source(source_t *source) {
  printf(" Source infos:\n\n");

  if (source->type == AUDIO) {
    double duration = (double)source->numsamples / source->sampling;

    printf("          type: audio data file\n");
    printf("      sampling: %d Hz\n", source->sampling);
    printf("      duration: %g\n", duration);

  } else {
    printf("          type: sine wave\n");
    printf("     frequency: %g Hz\n", source->data[0]);
  }

  printf("    position x: %g\n", source->posx);
  printf("    position y: %g\n", source->posy);
  printf("    position z: %g\n\n", source->posy);
}

void print_output(output_t *output) {
  switch (output->source) {
  case PRESSURE:
    printf("      pressure: ");
    break;
  case VELOCITYX:
    printf("    velocity X: ");
    break;
  case VELOCITYY:
    printf("    velocity Y: ");
    break;
  case VELOCITYZ:
    printf("    velocity Z: ");
    break;

  default:
    break;
  }

  switch (output->type) {
  case ALL:
    printf("complete dump");
    break;
  case CUTX:
    printf("cut along the x axis at %g", output->posx);
    break;
  case CUTY:
    printf("cut along the y axis at %g", output->posy);
    break;
  case CUTZ:
    printf("cut along the z axis at %g", output->posz);
    break;
  case POINT:
    printf("single point at %g %g %g", output->posx, output->posy,
           output->posz);
    break;

  default:
    break;
  }

  printf(" to file %s\n", output->filename);
}

/******************************************************************************
 * Data functions                                                             *
 ******************************************************************************/
/*
data_t *allocate_data(grid_t *grid) {
  size_t numnodes = NUMNODESTOT(*grid);
  if (numnodes <= 0) {
    DEBUG_PRINTF("Invalid number of nodes (%lu)", numnodes);
    return NULL;
  }

  data_t *data;
  if ((data = malloc(sizeof(data_t))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    free(data);
    return NULL;
  }

  if ((data->vals = malloc(numnodes * sizeof(double))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    free(data->vals);
    free(data);
    return NULL;
  }

  data->grid = *grid;

  return data;
}
*/
data_rank_t *allocate_data_rank(grid_rank_t *grid_rank) {
  size_t numnodes = NUMNODESTOT(*grid_rank);
  if (numnodes <= 0) {
    DEBUG_PRINTF("Invalid number of nodes (%lu)", numnodes);
    return NULL;
  }

  data_rank_t *data_rank;
  if ((data_rank= malloc(sizeof(data_rank_t))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    free(data_rank);
    return NULL;
  }

  if ((data_rank->vals = malloc(numnodes * sizeof(double))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    free(data_rank->vals);
    free(data_rank);
    return NULL;
  }
  data_rank->border_vals = (double **)malloc(6 * sizeof(double *));
  
  if(data_rank->border_vals == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    free(data_rank->border_vals);
    //add free memory for *border_vals
    free(data_rank);
    return NULL;
  }
  for (int i = 0; i < 6; i++) {
        if((data_rank->border_vals[i] = (double*)malloc(numnodes * sizeof(double))) == NULL) {
          DEBUG_PRINT("Failed to allocate memory");
          return NULL;
        };
    }
  
  
  

  data_rank->grid = *grid_rank;

  return data_rank;
}

void fill_data(data_t *data, double value) {
  if (data == NULL) {
    DEBUG_PRINT("Invalid NULL data");
    return;
  }

  for (int m = 0; m < NUMNODESX(data); m++) {
    for (int n = 0; n < NUMNODESY(data); n++) {
      for (int p = 0; p < NUMNODESZ(data); p++) {
        SETVALUE(data, m, n, p, value);
      }
    }
  }
}

void fill_data_rank(simulation_data_rank_t *simdata_rank, data_rank_t *data_rank, double value) {
  if (data_rank == NULL) {
    DEBUG_PRINT("Invalid NULL data");
    return;
  }

  for (int m = simdata_rank -> grid.startm; m < simdata_rank -> grid.endm; m++) {
    for (int n = simdata_rank -> grid.startn; n < simdata_rank -> grid.endn; n++) {
      for (int p = simdata_rank -> grid.startp; p < simdata_rank -> grid.endp; p++) {
        setvalue(simdata_rank, data_rank, m, n, p, value);
      }
    }
  }
}

/******************************************************************************
 * Data file functions                                                        *
 ******************************************************************************/

FILE *create_datafile(grid_rank_t grid, char *filename) {
  if (filename == NULL) {
    DEBUG_PRINT("Invalid NULL filename");
    return NULL;
  }

  FILE *fp;
  if ((fp = fopen(filename, "wb")) == NULL) {
    DEBUG_PRINTF("Failed to open file '%s'", filename);
    return NULL;
  }

  if (fwrite(&grid.numnodesx, sizeof(int), 1, fp) != 1 ||
      fwrite(&grid.numnodesy, sizeof(int), 1, fp) != 1 ||
      fwrite(&grid.numnodesz, sizeof(int), 1, fp) != 1 ||
      fwrite(&grid.xmin, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.xmax, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.ymin, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.ymax, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.zmin, sizeof(double), 1, fp) != 1 ||
      fwrite(&grid.zmax, sizeof(double), 1, fp) != 1) {

    DEBUG_PRINTF("Failed to write header of file '%s'", filename);
    fclose(fp);
    return NULL;
  }

  return fp;
}

FILE *open_datafile(grid_rank_t *grid, int *numsteps, char *filename) {
  if (grid == NULL || filename == NULL) {
    DEBUG_PRINT("Invalid NULL grid or filename");
    return NULL;
  }

  FILE *fp;
  if ((fp = fopen(filename, "rb")) == NULL) {
    DEBUG_PRINTF("Failed to open file '%s'", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  size_t file_size = ftell(fp);
  rewind(fp);

  if (fread(&grid->numnodesx, sizeof(int), 1, fp) != 1 ||
      fread(&grid->numnodesy, sizeof(int), 1, fp) != 1 ||
      fread(&grid->numnodesz, sizeof(int), 1, fp) != 1 ||
      fread(&grid->xmin, sizeof(double), 1, fp) != 1 ||
      fread(&grid->xmax, sizeof(double), 1, fp) != 1 ||
      fread(&grid->ymin, sizeof(double), 1, fp) != 1 ||
      fread(&grid->ymax, sizeof(double), 1, fp) != 1 ||
      fread(&grid->zmin, sizeof(double), 1, fp) != 1 ||
      fread(&grid->zmax, sizeof(double), 1, fp) != 1) {

    DEBUG_PRINTF("Failed to read header of file '%s'", filename);
    fclose(fp);
    return NULL;
  }

  size_t numnodestot =
      (size_t)grid->numnodesx * grid->numnodesy * grid->numnodesz;

  size_t values_size = numnodestot * sizeof(double);
  size_t stepindex_size = sizeof(int);
  size_t timestamp_size = sizeof(double);
  size_t header_size = 6 * sizeof(double) + 3 * sizeof(int);

  size_t onetimestep_size = values_size + stepindex_size + timestamp_size;
  size_t alltimestep_size = file_size - header_size;

  if (alltimestep_size % onetimestep_size != 0) {
    DEBUG_PRINTF("Data size is inconsistent with number of nodes (%lu, %lu)",
                 alltimestep_size, onetimestep_size);

    fclose(fp);
    return NULL;
  }

  if (numsteps != NULL) {
    *numsteps = (alltimestep_size / onetimestep_size);
  }

  return fp;
}

data_rank_t *read_data(FILE *fp, grid_rank_t *grid, int *step, double *time) {
  if (fp == NULL) {
    DEBUG_PRINT("Invalid NULL file pointer");
    return NULL;
  }

  double ltime;
  int lstep;

  size_t numnodes = NUMNODESTOT(*grid);

  data_rank_t *data;
  if ((data = allocate_data_rank(grid)) == NULL) {
    DEBUG_PRINT("Failed to allocate data");
    return NULL;
  }

  if (fread(&lstep, sizeof(int), 1, fp) != 1 ||
      fread(&ltime, sizeof(double), 1, fp) != 1 ||
      fread(data->vals, sizeof(double), numnodes, fp) != numnodes) {
    DEBUG_PRINT("Failed to read data");
    free(data);
    return NULL;
  }

  if (step != NULL)
    *step = lstep;
  if (time != NULL)
    *time = ltime;

  return data;
}

int write_data(FILE *fp, data_rank_t *data_rank, int step, double time) {
  if (fp == NULL || data_rank == NULL || data_rank->vals == NULL) {
    DEBUG_PRINT("Invalid NULL data or file pointer");
    return 1;
  }

  size_t numnodes = NUMNODESTOT(data_rank->grid);
  if (numnodes <= 0) {
    DEBUG_PRINTF("Invalid number of nodes (%lu)", numnodes);
    return 1;
  }

  if (fwrite(&step, sizeof(int), 1, fp) != 1 ||
      fwrite(&time, sizeof(double), 1, fp) != 1 ||
      fwrite(data_rank->vals, sizeof(double), numnodes, fp) != numnodes) {
    DEBUG_PRINT("Failed to write data");
    return 1;
  }

  return 0;
}

/******************************************************************************
 * Output file functions                                                      *
 ******************************************************************************/

int write_output(simulation_data_rank_t *simdata_rank, output_t *output, data_rank_t *data_rank, int step, double time) {
  if (output == NULL || data_rank == NULL) {
    DEBUG_PRINT("NULL pointer passed as argument");
    return 1;
  }

  output_type_t type = output->type;

  if (type == ALL) {
    return write_data(output->fp, data_rank, step, time);
  }

  int m, n, p;
  closest_index(&data_rank->grid, output->posx, output->posy, output->posz, &m, &n,
                &p);

  int startm = (type == CUTX || type == POINT) ? m : 0;
  int startn = (type == CUTY || type == POINT) ? n : 0;
  int startp = (type == CUTZ || type == POINT) ? p : 0;

  int endm = (type == CUTX || type == POINT) ? m + 1 : NUMNODESX(data_rank);
  int endn = (type == CUTY || type == POINT) ? n + 1 : NUMNODESY(data_rank);
  int endp = (type == CUTZ || type == POINT) ? p + 1 : NUMNODESZ(data_rank);

  data_rank_t *tmpdata = allocate_data_rank(&output->grid);

  for (m = startm; m < endm; m++) {
    for (n = startn; n < endn; n++) {
      for (p = startp; p < endp; p++) {
        int tmpm = m - startm;
        int tmpn = n - startn;
        int tmpp = p - startp;

        setvalue(simdata_rank, tmpdata, tmpm, tmpn, tmpp, getvalue(simdata_rank, data_rank, m, n, p));
      }
    }
  }

  int writeok = (write_data(output->fp, tmpdata, step, time) == 0);

  free(tmpdata->vals);
  free(tmpdata);

  if (writeok == 0) {
    DEBUG_PRINT("Failed to write output data");
    return 1;
  }

  return 0;
}

int open_outputfile(output_t *output, grid_rank_t *simgrid_rank) {
  if (output == NULL || simgrid_rank == NULL) {
    DEBUG_PRINT("Invalid NULL pointer in argment");
    return 1;
  }

  grid_rank_t grid;

  output_type_t type = output->type;

  grid.numnodesx = (type == POINT || type == CUTX) ? 1 : simgrid_rank->numnodesx;
  grid.numnodesy = (type == POINT || type == CUTY) ? 1 : simgrid_rank->numnodesy;
  grid.numnodesz = (type == POINT || type == CUTZ) ? 1 : simgrid_rank->numnodesz;

  grid.xmin = (type == POINT || type == CUTX) ? output->posx : simgrid_rank->xmin;
  grid.xmax = (type == POINT || type == CUTX) ? output->posx : simgrid_rank->xmax;

  grid.ymin = (type == POINT || type == CUTY) ? output->posy : simgrid_rank->ymin;
  grid.ymax = (type == POINT || type == CUTY) ? output->posy : simgrid_rank->ymax;

  grid.zmin = (type == POINT || type == CUTZ) ? output->posz : simgrid_rank->zmin;
  grid.zmax = (type == POINT || type == CUTZ) ? output->posz : simgrid_rank->zmax;

  FILE *fp;
  if ((fp = create_datafile(grid, output->filename)) == NULL) {
    DEBUG_PRINTF("Failed to open output file: '%s'", output->filename);
    return 1;
  }

  output->grid = grid;
  output->fp = fp;

  return 0;
}

/******************************************************************************
 * Parameter file functions                                                   *
 ******************************************************************************/

int read_audiosource(char *filename, source_t *source) {
  FILE *fp;
  if ((fp = fopen(filename, "rb")) == NULL) {
    DEBUG_PRINTF("Could not open source file '%s'", filename);
    return 1;
  }

  fseek(fp, 0, SEEK_END);
  size_t filesize = ftell(fp);
  rewind(fp);

  int numsamples = (filesize - sizeof(int)) / sizeof(double);

  int sampling;
  if (fread(&sampling, sizeof(int), 1, fp) != 1) {
    DEBUG_PRINT("Failed to read source data");
    fclose(fp);
    return 1;
  }

  double *data;
  if ((data = malloc(numsamples * sizeof(double))) == NULL) {
    DEBUG_PRINT("Failed to allocate memory for source data");
    return 1;
  }

  int readok = (fread(data, sizeof(double), numsamples, fp) == numsamples);

  fclose(fp);

  if (readok == 0) {
    DEBUG_PRINT("Failed to read source data");
    return 1;
  }

  source->data = data;
  source->numsamples = numsamples;
  source->sampling = sampling;

  return 0;
}

int read_outputparam(FILE *fp, output_t *output) {
  if (fp == NULL || output == NULL) {
    DEBUG_PRINT("NULL passed as argement");
    return 1;
  }

  char typekeyword[BUFSZ_SMALL];
  char sourcekeyword[BUFSZ_SMALL];
  char filename[BUFSZ_LARGE];

  double posxyz[3] = {0.0, 0.0, 0.0};

  if (fscanf(fp, BUFFMT_SMALL, typekeyword) != 1 ||
      fscanf(fp, BUFFMT_SMALL, sourcekeyword) != 1 ||
      fscanf(fp, BUFFMT_LARGE, filename) != 1) {

    DEBUG_PRINT("Failed to read an output parameter");
    return 1;
  }

  output_type_t type = CUTX;
  while (type < OUTPUT_TYPE_END &&
         strcmp(output_type_keywords[type], typekeyword) != 0) {
    type++;
  }

  if (type == OUTPUT_TYPE_END) {
    DEBUG_PRINTF("Invalid keyword: '%s'", typekeyword);
    return 1;
  }

  output_source_t source = PRESSURE;
  while (source < OUTPUT_SOURCE_END &&
         strcmp(output_source_keywords[source], sourcekeyword) != 0) {
    source++;
  }

  if (source == OUTPUT_SOURCE_END) {
    DEBUG_PRINTF("Invalid keyword: '%s'", sourcekeyword);
    return 1;
  }

  int readok = 1;
  switch (type) {
  case CUTX:
    readok = (fscanf(fp, "%lf", &posxyz[0]) == 1);
    break;
  case CUTY:
    readok = (fscanf(fp, "%lf", &posxyz[1]) == 1);
    break;
  case CUTZ:
    readok = (fscanf(fp, "%lf", &posxyz[2]) == 1);
    break;
  case ALL:
    break;

  case POINT:
    readok =
        (fscanf(fp, "%lf %lf %lf", &posxyz[0], &posxyz[1], &posxyz[2]) == 3);
    break;

  default:
    break;
  }

  if (readok == 0) {
    DEBUG_PRINT("Failed to read an output parameter");
    return 1;
  }

  output->filename = copy_string(filename);
  output->type = type;
  output->source = source;
  output->posx = posxyz[0];
  output->posy = posxyz[1];
  output->posz = posxyz[2];

  return 0;
}

int read_sourceparam(FILE *fp, source_t *source) {
  char typekeyword[BUFSZ_SMALL];
  char filename[BUFSZ_LARGE];

  double freq, posx, posy, posz;

  if (fscanf(fp, BUFFMT_SMALL, typekeyword) != 1) {
    DEBUG_PRINT("Failed to read the source parameter");
    return 1;
  }

  source_type_t type = SINE;
  while (type < SOURCE_TYPE_END &&
         strcmp(source_type_keywords[type], typekeyword) != 0) {
    type++;
  }

  if (type == SOURCE_TYPE_END) {
    DEBUG_PRINTF("Invalid keyword: '%s'", typekeyword);
    return 1;
  }

  int readok = 1;
  switch (type) {
  case SINE:
    readok = (fscanf(fp, "%lf", &freq) == 1);
    break;
  case AUDIO:
    readok = (fscanf(fp, BUFFMT_LARGE, filename) == 1);
    break;

  default:
    break;
  }

  if (readok == 0 || fscanf(fp, "%lf %lf %lf", &posx, &posy, &posz) != 3) {
    DEBUG_PRINT("Failed to read the source parameter");
    return 1;
  }

  switch (type) {
  case AUDIO:
    read_audiosource(filename, source);
    break;
  case SINE: {
    if ((source->data = malloc(sizeof(double))) == NULL) {
      DEBUG_PRINT("Failed to allocate memory");
      return 1;
    }

    source->data[0] = freq;
    source->numsamples = 1;

    break;
  }

  default:
    break;
  }

  source->type = type;
  source->posx = posx;
  source->posy = posy;
  source->posz = posz;

  return 0;
}

int read_paramfile(parameters_t *params, const char *filename) {
  if (params == NULL || filename == NULL) {
    DEBUG_PRINT("Invalid print_out params or filename");
    return 1;
  }

  int outrate, numoutputs = 0;

  double dx, dt, maxt;

  char cin_filename[BUFSZ_LARGE];
  char rhoin_filename[BUFSZ_LARGE];

  source_t source;
  output_t *outputs = NULL;

  if ((outputs = malloc(sizeof(output_t) * MAX_OUTPUTS)) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return 1;
  }

  FILE *fp;
  if ((fp = fopen(filename, "r")) == NULL) {
    DEBUG_PRINTF("Could not open parameter file '%s'", filename);
    return 1;
  }

  int readok =
      ((fscanf(fp, "%lf", &dx) == 1) && (fscanf(fp, "%lf", &dt) == 1) &&
       (fscanf(fp, "%lf", &maxt) == 1) && (fscanf(fp, "%d", &outrate) == 1) &&
       (fscanf(fp, BUFFMT_LARGE, cin_filename) == 1) &&
       (fscanf(fp, BUFFMT_LARGE, rhoin_filename) == 1));

  readok = (readok != 0 && read_sourceparam(fp, &source) == 0 &&
            fscanf(fp, " ") == 0);

  while (readok != 0 && numoutputs < MAX_OUTPUTS && feof(fp) == 0) {
    readok = (read_outputparam(fp, &outputs[numoutputs++]) == 0 &&
              fscanf(fp, " ") == 0);
  }

  fclose(fp);

  if (readok == 0) {
    DEBUG_PRINT("Failed to read parameter file");
    free(outputs);
    return 1;
  }

  if (numoutputs == 0) {
    free(outputs);
    outputs = NULL;

  } else if ((outputs = realloc(outputs, sizeof(output_t) * numoutputs)) ==
             NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return 1;
  }

  params->dx = dx;
  params->dt = dt;
  params->maxt = maxt;
  params->outrate = outrate;
  params->cin_filename = copy_string(cin_filename);
  params->rhoin_filename = copy_string(rhoin_filename);
  params->source = source;
  params->numoutputs = numoutputs;
  params->outputs = outputs;

  return 0;
}

/******************************************************************************
 * Simulation related functions                                               *
 ******************************************************************************/
//NEEDS TO BE MODIFIED MAYBE? 
int interpolate_inputmaps(simulation_data_rank_t *simdata, grid_rank_t *simgrid,
                          data_rank_t *cin, data_rank_t *rhoin) {
  if (simdata == NULL || cin == NULL) {
    DEBUG_PRINT("Invalid NULL simdata or cin");
    return 1;
  }

  if ((simdata->c = allocate_data_rank(simgrid)) == NULL ||
      (simdata->rho = allocate_data_rank(simgrid)) == NULL ||
      (simdata->rhohalf = allocate_data_rank(simgrid)) == NULL) {
    DEBUG_PRINT("Failed to allocate memory");
    return 1;
  }

  double dx = simdata->params.dx;
  double dxd2 = simdata->params.dx / 2;

  for (int m = simdata -> grid.startm; m < simdata->grid.endm; m++) {
    for (int n = simdata -> grid.startn; n < simdata -> grid.endn; n++) {
      for (int p = simdata -> grid.startp; p < simdata -> grid.endp; p++) {

        double x = m * dx;
        double y = n * dx;
        double z = p * dx;

        int mc, nc, pc;
        closest_index(&cin->grid, x, y, z, &mc, &nc, &pc);

        setvalue(simdata, simdata->c, m, n, p, getvalue(simdata, cin, mc, nc, pc));
        setvalue(simdata, simdata->rho, m, n, p, getvalue(simdata, rhoin, mc, nc, pc));

        x += dxd2;
        y += dxd2;
        z += dxd2;

        closest_index(&rhoin->grid, x, y, z, &mc, &nc, &pc);
        setvalue(simdata, simdata->rhohalf, m, n, p, getvalue(simdata, rhoin, mc, nc, pc));
      }
    }
  }

  return 0;
}

void apply_source(simulation_data_rank_t *simdata_rank, int step) {
  source_t *source = &simdata_rank->params.source;

  double posx = source->posx;
  double posy = source->posy;
  double posz = source->posz;

  double t = step * simdata_rank->params.dt;

  int m, n, p;
  closest_index(&simdata_rank->pold->grid, posx, posy, posz, &m, &n, &p);

  if (source->type == SINE) {
    double freq = source->data[0];
    
    setvalue(simdata_rank,simdata_rank->pold, m, n, p, sin(2 * M_PI * freq * t));
    
  } else if (source->type == AUDIO) {
    int sample = MIN((int)(t * source->sampling), source->numsamples - 1);
    
    setvalue(simdata_rank,simdata_rank->pold, m, n, p, simdata_rank->params.source.data[sample]);
    
  }
  
}

void update_pressure(simulation_data_rank_t *simdata_rank) {
  
  const double dtdx = simdata_rank->params.dt / simdata_rank->params.dx;
  MPI_Request recv_req[3];
  MPI_Request send_req[3];
  
  
  //start of RECEIVE process
  //We store the falttened matrix at the address of border_val[0] as we only need one face.
  // check tags, dims, dest and sources
  //DEBUG_PRINT(" 1 Start of receive pressure");
  MPI_Irecv(simdata_rank -> vxold -> border_vals[LEFT], NUMNODESY(simdata_rank)*NUMNODESZ(simdata_rank), MPI_DOUBLE, neighbors[RIGHT], 0, MPI_COMM_WORLD, &recv_req[0]);
  
  MPI_Irecv(simdata_rank -> vyold -> border_vals[DOWN], NUMNODESZ(simdata_rank)*NUMNODESX(simdata_rank), MPI_DOUBLE, neighbors[DOWN], 1, MPI_COMM_WORLD, &recv_req[1]);
  
  MPI_Irecv(simdata_rank -> vzold -> border_vals[IN], NUMNODESX(simdata_rank)*NUMNODESY(simdata_rank), MPI_DOUBLE, neighbors[IN], 2, MPI_COMM_WORLD, &recv_req[2]);
  //DEBUG_PRINT(" 2 End receive pressure");
  
  //Memory allocation for SEND process
  double* data_out   = (double *) malloc(NUMNODESY(simdata_rank)*NUMNODESZ(simdata_rank)*sizeof(double ));
  double* data_right = (double *) malloc(NUMNODESZ(simdata_rank)*NUMNODESX(simdata_rank)*sizeof(double ));
  double* data_up    = (double *) malloc(NUMNODESX(simdata_rank)*NUMNODESY(simdata_rank)*sizeof(double ));

  if(data_out == NULL || data_right == NULL || data_up == NULL){ 
    DEBUG_PRINT("Failed to allocate data");
    exit(0);  
    }
  
  //Storage pnew on IN face 
  //MAYBE simdata_rank -> grid.startm + 1 ??
  for (int m = simdata_rank -> grid.startm ; m < simdata_rank -> grid.endm ; m++) {
    for (int n = simdata_rank -> grid.startn ; n < simdata_rank -> grid.endn ; n++) {
        int p = simdata_rank -> grid.endp; //?
        data_out[m * NUMNODESY(simdata_rank) + n] = getvalue(simdata_rank, simdata_rank -> vzold, m,n,p); 
    }
  }

  //Storage pnew on RIGHT face
  for (int p = simdata_rank -> grid.startp; p < simdata_rank -> grid.endp; p++) {
    for (int n = simdata_rank -> grid.startn; n < simdata_rank -> grid.endn; n++) {
        int m = simdata_rank -> grid.endm; //?
        data_right[p * NUMNODESY(simdata_rank) + n] = getvalue(simdata_rank, simdata_rank -> vxold, m,n,p);
    }
  }

  //Storage pnew on UP face
  for (int p = simdata_rank -> grid.startp; p < simdata_rank -> grid.endp; p++) {
    for (int m = simdata_rank -> grid.startn; m < simdata_rank -> grid.endn; m++) {
        int n = simdata_rank -> grid.endn; //?
        data_right[p * NUMNODESX(simdata_rank) + m] = getvalue(simdata_rank, simdata_rank -> vyold, m,n,p);
    }
  }
  
  //Start of Send data process
  // check tags, dims, dest and sources
  //DEBUG_PRINT(" 3 Start of send pressure");
  MPI_Isend(data_right, NUMNODESY(simdata_rank)*NUMNODESZ(simdata_rank), MPI_DOUBLE, neighbors[LEFT], 0, MPI_COMM_WORLD, &send_req[0]);

  MPI_Isend(data_up, NUMNODESZ(simdata_rank)*NUMNODESX(simdata_rank), MPI_DOUBLE, neighbors[UP], 1, MPI_COMM_WORLD, &send_req[1]);
  
  MPI_Isend(data_out, NUMNODESX(simdata_rank)*NUMNODESY(simdata_rank), MPI_DOUBLE, neighbors[OUT], 2, MPI_COMM_WORLD, &send_req[2]);
  
  
  //DEBUG_PRINT(" 4 End of send pressure");

  //update interior faces
  for (int m = simdata_rank -> grid.startm + 1; m < simdata_rank -> grid.endm; m++) {
    for (int n = simdata_rank -> grid.startn + 1; n < simdata_rank -> grid.endn; n++) {
      for (int p = simdata_rank -> grid.startp + 1; p < simdata_rank -> grid.endp; p++) {
        
        double c = getvalue(simdata_rank, simdata_rank -> c, m, n, p);
        double rho = getvalue(simdata_rank, simdata_rank->rho, m, n, p);

        double rhoc2dtdx = rho * c * c * dtdx;

        double dvx = getvalue(simdata_rank, simdata_rank->vxold, m, n, p);
        double dvy = getvalue(simdata_rank, simdata_rank->vyold, m, n, p);
        double dvz = getvalue(simdata_rank, simdata_rank->vzold, m, n, p);

        dvx -= m > 0 ? getvalue(simdata_rank, simdata_rank->vxold, m - 1, n, p) : 0.0;
        dvy -= n > 0 ? getvalue(simdata_rank, simdata_rank->vyold, m, n - 1, p) : 0.0;
        dvz -= p > 0 ? getvalue(simdata_rank, simdata_rank->vzold, m, n, p - 1) : 0.0;

        double prev_p = getvalue(simdata_rank, simdata_rank->pold, m, n, p);
        
        setvalue(simdata_rank, simdata_rank->pnew, m, n, p,
                 (prev_p - rhoc2dtdx * (dvx + dvy + dvz)));
      }
    }
  }
  
  //Wait for all receivers
  //DEBUG_PRINT(" 5 Start of Wait process receive");
  MPI_Waitall(3, recv_req, MPI_STATUS_IGNORE);
  //DEBUG_PRINT(" 6 End of Wait process receive");
  //Start the computation on subdomain boundaries using **ghostvals

  //Check loop index to not pass over repeated cells (i.e. "-1" and "+1" in the following 3 double loops)

  //data_vz
  for (int m = simdata_rank -> grid.startm ; m < simdata_rank -> grid.endm ; m++) {
    for (int n = simdata_rank -> grid.startn ; n < simdata_rank -> grid.endn ; n++) {
        
        int p = simdata_rank -> grid.startp; 
        
        double c = getvalue(simdata_rank, simdata_rank -> c, m, n, p);
        double rho = getvalue(simdata_rank, simdata_rank->rho, m, n, p);

        double rhoc2dtdx = rho * c * c * dtdx;

        double dvx = getvalue(simdata_rank, simdata_rank->vxold, m, n, p);
        double dvy = getvalue(simdata_rank, simdata_rank->vyold, m, n, p);
        double dvz = getvalue(simdata_rank, simdata_rank->vzold, m, n, p);
        
        dvx -= m > 0 ? getvalue(simdata_rank, simdata_rank->vxold, m - 1, n, p) : 0.0;
        dvy -= n > 0 ? getvalue(simdata_rank, simdata_rank->vyold, m, n - 1, p) : 0.0;
        dvz -= p > 0 ? simdata_rank-> vzold -> border_vals[IN][m * NUMNODESY(simdata_rank) + n] : 0.0; // = dvz[m,n,p - 1] ?

        double prev_p = getvalue(simdata_rank, simdata_rank->pold, m, n, p);

        setvalue(simdata_rank, simdata_rank->pnew, m, n, p,
                 (prev_p - rhoc2dtdx * (dvx + dvy + dvz)));
    }    
  }

  //data_vy
  for (int m = simdata_rank -> grid.startm ; m < simdata_rank -> grid.endm ; m++) {
    for (int p = simdata_rank -> grid.startp + 1 ; p < simdata_rank -> grid.endp - 1; p++) {
        
        int n = simdata_rank -> grid.startn; 

        double c = getvalue(simdata_rank, simdata_rank -> c, m, n, p);
        double rho = getvalue(simdata_rank, simdata_rank->rho, m, n, p);

        double rhoc2dtdx = rho * c * c * dtdx;

        double dvx = getvalue(simdata_rank, simdata_rank->vxold, m, n, p);
        double dvy = getvalue(simdata_rank, simdata_rank->vyold, m, n, p);
        double dvz = getvalue(simdata_rank, simdata_rank->vzold, m, n, p);
        
        dvx -= m > 0 ? getvalue(simdata_rank, simdata_rank->vxold, m - 1, n, p) : 0.0;
        dvy -= n > 0 ? simdata_rank-> vyold -> border_vals[DOWN][m * NUMNODESZ(simdata_rank) + p] : 0.0;
        dvz -= p > 0 ? getvalue(simdata_rank, simdata_rank->vzold, m, n, p - 1) : 0.0;

        double prev_p = getvalue(simdata_rank, simdata_rank->pold, m, n, p);
        setvalue(simdata_rank, simdata_rank->pnew, m, n, p,
                 (prev_p - rhoc2dtdx * (dvx + dvy + dvz)));
    } 
  }

   //data_vx
   for (int n = simdata_rank -> grid.startn + 1; n < simdata_rank -> grid.endn - 1; n++) {
    for (int p = simdata_rank -> grid.startp + 1; p < simdata_rank -> grid.endp -  1; p++) {
      
        int m = simdata_rank -> grid.startm; 

        double c = getvalue(simdata_rank, simdata_rank -> c, m, n, p);
        double rho = getvalue(simdata_rank, simdata_rank->rho, m, n, p);

        double rhoc2dtdx = rho * c * c * dtdx;

        double dvx = getvalue(simdata_rank, simdata_rank->vxold, m, n, p);
        double dvy = getvalue(simdata_rank, simdata_rank->vyold, m, n, p);
        double dvz = getvalue(simdata_rank, simdata_rank->vzold, m, n, p);
        
        dvx -= m > 0 ? simdata_rank-> vxold -> border_vals[LEFT][n * NUMNODESZ(simdata_rank) + p] : 0.0;
        dvy -= n > 0 ? getvalue(simdata_rank, simdata_rank->vyold, m, n - 1, p) : 0.0;
        dvz -= p > 0 ? getvalue(simdata_rank, simdata_rank->vzold, m, n, p - 1) : 0.0;

        double prev_p = getvalue(simdata_rank, simdata_rank->pold, m, n, p);

        setvalue(simdata_rank, simdata_rank->pnew, m, n, p,
                 (prev_p - rhoc2dtdx * (dvx + dvy + dvz)));
        
    }
  }
  //DEBUG_PRINT(" 7 Start of wait send process");
  MPI_Waitall(3, send_req, MPI_STATUS_IGNORE);
  //DEBUG_PRINT(" 8 End of wait send process");
}

void update_velocities(simulation_data_rank_t *simdata_rank){
  const double dtdx = simdata_rank->params.dt / simdata_rank->params.dx;

  MPI_Request send_req[3];
  MPI_Request rec_req[3];
    
  // check tags, dims, dest and sources
  //DEBUG_PRINT(" 9 Start of receive vel");
  MPI_Irecv(simdata_rank -> pnew -> border_vals[RIGHT], NUMNODESY(simdata_rank)*NUMNODESZ(simdata_rank), MPI_DOUBLE, neighbors[LEFT], 4, MPI_COMM_WORLD, &rec_req[0]);
  
  MPI_Irecv(simdata_rank -> pnew -> border_vals[OUT], NUMNODESY(simdata_rank)*NUMNODESX(simdata_rank), MPI_DOUBLE, neighbors[OUT], 5, MPI_COMM_WORLD, &rec_req[1]);

  MPI_Irecv(simdata_rank -> pnew -> border_vals[DOWN], NUMNODESX(simdata_rank)*NUMNODESZ(simdata_rank), MPI_DOUBLE, neighbors[DOWN], 6, MPI_COMM_WORLD, &rec_req[2]);
  //DEBUG_PRINT(" 10 End receive vel");
  //Memory allocation for SEND process
  
  double* data_left    = (double *) malloc(NUMNODESZ(simdata_rank)*NUMNODESX(simdata_rank)*sizeof(double ));
  double* data_out     = (double *) malloc(NUMNODESY(simdata_rank)*NUMNODESX(simdata_rank)*sizeof(double ));
  double* data_down    = (double *) malloc( NUMNODESX(simdata_rank)*NUMNODESZ(simdata_rank)*sizeof(double ));
  
  if(data_out == NULL || data_left == NULL || data_down == NULL){ 
    DEBUG_PRINT("Failed to allocate data");
    exit(0);  
  }

  for(int m = simdata_rank -> grid.startm; m < simdata_rank -> grid.endm; m++){
    for(int n = simdata_rank -> grid.startn; n < simdata_rank -> grid.endn; n++){
      int p = simdata_rank -> grid.startp;
      data_out[m*NUMNODESY(simdata_rank) + n] = getvalue(simdata_rank, simdata_rank -> pnew, m , n, p);
    }
  }
  for(int m = simdata_rank -> grid.startm; m < simdata_rank -> grid.endm; m++){
    for(int p = simdata_rank -> grid.startp; p < simdata_rank -> grid.endp; p++){
      int n = simdata_rank -> grid.startn;
      data_down[m*NUMNODESZ(simdata_rank) + p] = getvalue(simdata_rank, simdata_rank -> pnew, m , n, p);
    }
  }
  for(int p = simdata_rank -> grid.startp; p < simdata_rank -> grid.endp; p++){
    for(int n = simdata_rank -> grid.startn; n < simdata_rank -> grid.endn; n++){
      int m = simdata_rank -> grid.startm;
      data_left[p*NUMNODESZ(simdata_rank) + n] = getvalue(simdata_rank, simdata_rank -> pnew, m , n, p);
    }
  }
  // check tags, dims, dest and sources
  //DEBUG_PRINT(" 11 Start of send vel");
  MPI_Isend(data_left, NUMNODESZ(simdata_rank)*NUMNODESX(simdata_rank), MPI_DOUBLE, neighbors[RIGHT], 4, MPI_COMM_WORLD, &send_req[0]);

  MPI_Isend(data_out, NUMNODESY(simdata_rank)*NUMNODESX(simdata_rank), MPI_DOUBLE, neighbors[IN], 5, MPI_COMM_WORLD, &send_req[1]);
  
  MPI_Isend(data_down,  NUMNODESX(simdata_rank)*NUMNODESZ(simdata_rank), MPI_DOUBLE, neighbors[UP], 6, MPI_COMM_WORLD, &send_req[2]);

  
  //DEBUG_PRINT(" 12 End of send vel");
  
  //check indexs
  for (int m = simdata_rank -> grid.startm + 1; m < simdata_rank -> grid.endm ; m++) {
    for (int n = simdata_rank -> grid.startn + 1; n < simdata_rank -> grid.endn; n++) {
      for (int p = simdata_rank -> grid.startp + 1; p < simdata_rank -> grid.endp; p++) {
        int mp1 = MIN(simdata_rank -> grid.endm - 1, m + 1);
        int np1 = MIN(simdata_rank -> grid.endn - 1, n + 1);
        int pp1 = MIN(simdata_rank -> grid.endp - 1, p + 1);

        double dtdxrho = dtdx / getvalue(simdata_rank, simdata_rank->rhohalf, m, n, p);

        double p_mnq = getvalue(simdata_rank,simdata_rank->pnew, m, n, p);

        double dpx = getvalue(simdata_rank,simdata_rank->pnew, mp1, n, p) - p_mnq;
        double dpy = getvalue(simdata_rank,simdata_rank->pnew, m, np1, p) - p_mnq;
        double dpz = getvalue(simdata_rank, simdata_rank->pnew, m, n, pp1) - p_mnq;

        double prev_vx = getvalue(simdata_rank,simdata_rank->vxold, m, n, p);
        double prev_vy = getvalue(simdata_rank,simdata_rank->vyold, m, n, p);
        double prev_vz = getvalue(simdata_rank,simdata_rank->vzold, m, n, p);

        setvalue(simdata_rank,simdata_rank->vxnew, m, n, p, prev_vx - dtdxrho * dpx);
        setvalue(simdata_rank,simdata_rank->vynew, m, n, p, prev_vy - dtdxrho * dpy);
        setvalue(simdata_rank,simdata_rank->vznew, m, n, p, prev_vz - dtdxrho * dpz);
      }
    }
  }
  //DEBUG_PRINT(" 13 Start of wait receive vel");
  MPI_Waitall(3, rec_req, MPI_STATUS_IGNORE);
  //DEBUG_PRINT(" 14 End of wait receive vel");
  //Start of computation on bondary faces
  // IN face
  for(int m = simdata_rank -> grid.startm; m < simdata_rank -> grid.endm; m++){
    for(int n = simdata_rank -> grid.startn; n < simdata_rank -> grid.endn; n++){
        int mp1 = MIN(simdata_rank -> grid.endm - 1, m + 1);
        int np1 = MIN(simdata_rank -> grid.endn - 1, n + 1);
        int p = simdata_rank -> grid.endp;
        
        double dtdxrho = dtdx / getvalue(simdata_rank, simdata_rank->rhohalf, m, n, p);

        double p_mnq = getvalue(simdata_rank,simdata_rank->pnew, m, n, p);

        double dpx = getvalue(simdata_rank,simdata_rank->pnew, mp1, n, p) - p_mnq;
        double dpy = getvalue(simdata_rank,simdata_rank->pnew, m, np1, p) - p_mnq;
        double dpz = simdata_rank -> pnew -> border_vals[OUT][m * NUMNODESY(simdata_rank) + n] - p_mnq;

        double prev_vx = getvalue(simdata_rank,simdata_rank->vxold, m, n, p);
        double prev_vy = getvalue(simdata_rank,simdata_rank->vyold, m, n, p);
        double prev_vz = getvalue(simdata_rank,simdata_rank->vzold, m, n, p);

        setvalue(simdata_rank,simdata_rank->vxnew, m, n, p, prev_vx - dtdxrho * dpx);
        setvalue(simdata_rank,simdata_rank->vynew, m, n, p, prev_vy - dtdxrho * dpy);
        setvalue(simdata_rank,simdata_rank->vznew, m, n, p, prev_vz - dtdxrho * dpz);
    }
  }

  //DOWN face
  for(int m = simdata_rank -> grid.startm; m < simdata_rank -> grid.endm; m++){
    for(int p = simdata_rank -> grid.startp + 1; p < simdata_rank -> grid.endp - 1; p++){
        int mp1 = MIN(simdata_rank -> grid.endm - 1, m + 1);
        int n = simdata_rank -> grid.endn;
        int pp1 = MIN(simdata_rank -> grid.endp - 1, p + 1);
        
        double dtdxrho = dtdx / getvalue(simdata_rank, simdata_rank->rhohalf, m, n, p);

        double p_mnq = getvalue(simdata_rank,simdata_rank->pnew, m, n, p);

        double dpx = getvalue(simdata_rank,simdata_rank->pnew, mp1, n, p) - p_mnq;
        double dpy = simdata_rank -> pnew -> border_vals[DOWN][m * NUMNODESZ(simdata_rank) + p] - p_mnq;
        double dpz = getvalue(simdata_rank,simdata_rank->pnew, m, n, pp1) - p_mnq;

        double prev_vx = getvalue(simdata_rank,simdata_rank->vxold, m, n, p);
        double prev_vy = getvalue(simdata_rank,simdata_rank->vyold, m, n, p);
        double prev_vz = getvalue(simdata_rank,simdata_rank->vzold, m, n, p);

        setvalue(simdata_rank,simdata_rank->vxnew, m, n, p, prev_vx - dtdxrho * dpx);
        setvalue(simdata_rank,simdata_rank->vynew, m, n, p, prev_vy - dtdxrho * dpy);
        setvalue(simdata_rank,simdata_rank->vznew, m, n, p, prev_vz - dtdxrho * dpz);
    }
  }
  //RIGHT face
  for(int n = simdata_rank -> grid.startn + 1; n < simdata_rank -> grid.endn - 1; n++){
    for(int p = simdata_rank -> grid.startp + 1; p < simdata_rank -> grid.endp - 1; p++){
        int m = simdata_rank -> grid.endm;
        int np1 = MIN(simdata_rank -> grid.endn - 1, n + 1);
        int pp1 = MIN(simdata_rank -> grid.endp - 1, p + 1);
        
        double dtdxrho = dtdx / getvalue(simdata_rank, simdata_rank->rhohalf, m, n, p);

        double p_mnq = getvalue(simdata_rank,simdata_rank->pnew, m, n, p);

        double dpx = simdata_rank -> pnew -> border_vals[LEFT][n * NUMNODESZ(simdata_rank) + p] - p_mnq;
        double dpy =  getvalue(simdata_rank,simdata_rank->pnew, m, np1, p) - p_mnq;
        double dpz = getvalue(simdata_rank,simdata_rank->pnew, m, n, pp1) - p_mnq;

        double prev_vx = getvalue(simdata_rank,simdata_rank->vxold, m, n, p);
        double prev_vy = getvalue(simdata_rank,simdata_rank->vyold, m, n, p);
        double prev_vz = getvalue(simdata_rank,simdata_rank->vzold, m, n, p);

        setvalue(simdata_rank,simdata_rank->vxnew, m, n, p, prev_vx - dtdxrho * dpx);
        setvalue(simdata_rank,simdata_rank->vynew, m, n, p, prev_vy - dtdxrho * dpy);
        setvalue(simdata_rank,simdata_rank->vznew, m, n, p, prev_vz - dtdxrho * dpz);
    }
  }
  //DEBUG_PRINT(" 15 Start of wait send process vel");
  MPI_Waitall(3, send_req, MPI_STATUS_IGNORE);
  //DEBUG_PRINT(" 16 End of wait send process vel");
}

void init_simulation(simulation_data_rank_t *simdata_rank, const char *params_filename) {
  //DEBUG_PRINT(" init_simulation");
  if (read_paramfile(&simdata_rank->params, params_filename) != 0) {
    printf("Failed to read parameters. Aborting...\n\n");
    exit(1);
  }
  
  grid_rank_t rhoin_grid;
  grid_rank_t cin_grid;
  grid_rank_t sim_grid;
  

  int rho_numstep;
  int c_numstep;

  FILE *rhofp =
      open_datafile(&rhoin_grid, &rho_numstep, simdata_rank->params.rhoin_filename);
  FILE *cfp =
      open_datafile(&cin_grid, &c_numstep, simdata_rank->params.cin_filename);

  if (rhofp == NULL || rho_numstep <= 0) {
    printf("Failed to open the density map file. Aborting...\n\n");
    exit(1);
  }

  if (cfp == NULL || c_numstep <= 0) {
    printf("Failed to open the speed map file. Aborting...\n\n");
    exit(1);
  }

  if (rhoin_grid.xmin != cin_grid.xmin || rhoin_grid.ymin != cin_grid.ymin ||
      rhoin_grid.zmin != cin_grid.zmin || rhoin_grid.xmax != cin_grid.xmax ||
      rhoin_grid.ymax != cin_grid.ymax || rhoin_grid.zmax != cin_grid.zmax) {
    printf("Grids for the density and speed are not the same. Aborting...\n\n");
    exit(1);
  }

  data_rank_t *rho_map = read_data(rhofp, &rhoin_grid, NULL, NULL);
  data_rank_t *c_map = read_data(cfp, &cin_grid, NULL, NULL);

  if (rho_map == NULL || c_map == NULL) {
    printf("Failed to read data from input maps. Aborting...\n\n");
    exit(1);
  }

  fclose(rhofp);
  fclose(cfp);

  sim_grid.xmin = rhoin_grid.xmin;
  sim_grid.xmax = rhoin_grid.xmax;
  sim_grid.ymin = rhoin_grid.ymin;
  sim_grid.ymax = rhoin_grid.ymax;
  sim_grid.zmin = rhoin_grid.zmin;
  sim_grid.zmax = rhoin_grid.zmax;
  
  //global numnodes 
  sim_grid.numnodesx =
      MAX(floor((sim_grid.xmax - sim_grid.xmin) / simdata_rank->params.dx), 1);
  sim_grid.numnodesy =
      MAX(floor((sim_grid.ymax - sim_grid.ymin) / simdata_rank->params.dx), 1);
  sim_grid.numnodesz =
      MAX(floor((sim_grid.zmax - sim_grid.zmin) / simdata_rank->params.dx), 1);
  //global x_min/x_max
  simdata_rank -> grid.xmin = rhoin_grid.xmin;
  simdata_rank -> grid.ymin = rhoin_grid.ymin;
  simdata_rank -> grid.zmin = rhoin_grid.zmin;

  simdata_rank -> grid.xmax = rhoin_grid.xmax;
  simdata_rank -> grid.ymax = rhoin_grid.ymax;
  simdata_rank -> grid.zmax = rhoin_grid.zmax;
  
  simdata_rank -> grid.numnodesx = sim_grid.numnodesx;
  simdata_rank -> grid.numnodesy = sim_grid.numnodesy;
  simdata_rank -> grid.numnodesz = sim_grid.numnodesz;

  simdata_rank -> grid.m_glob = sim_grid.numnodesx;
  simdata_rank -> grid.n_glob = sim_grid.numnodesy;
  simdata_rank -> grid.p_glob = sim_grid.numnodesz;

  simdata_rank -> grid.startm = (sim_grid.numnodesx*coords[0])/world_size;
  simdata_rank -> grid.startn = (sim_grid.numnodesy*coords[1])/world_size;
  simdata_rank -> grid.startp = (sim_grid.numnodesz*coords[2])/world_size;
  
  simdata_rank -> grid.endm = ((sim_grid.numnodesx*(coords[0] + 1)))/world_size - 1;
  simdata_rank -> grid.endn = ((sim_grid.numnodesy*(coords[1] + 1)))/world_size - 1;
  simdata_rank -> grid.endp = ((sim_grid.numnodesz*(coords[2] + 1)))/world_size - 1;
  
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  printf("numnodes = %d,rank = %d", simdata_rank->grid.numnodesx, rank);

  
  //TO BE CHANGED TO BE PARALLEL

  if (interpolate_inputmaps(simdata_rank, &sim_grid, c_map, rho_map) != 0) {
    printf(
        "Error while converting input map to simulation grid. Aborting...\n\n");
    exit(1);
  }

  if (simdata_rank->params.outrate > 0 && simdata_rank->params.outputs != NULL) {
    for (int i = 0; i < simdata_rank->params.numoutputs; i++) {
      char *outfilei = simdata_rank->params.outputs[i].filename;

      for (int j = 0; j < i; j++) {
        char *outfilej = simdata_rank->params.outputs[j].filename;

        if (strcmp(outfilei, outfilej) == 0) {
          printf("Duplicate output file: '%s'. Aborting...\n\n", outfilei);
          exit(1);
        }
      }
    }

    for (int i = 0; i < simdata_rank->params.numoutputs; i++) {
      output_t *output = &simdata_rank->params.outputs[i];

      if (open_outputfile(output, &sim_grid) != 0) {
        printf("Failed to open output file: '%s'. Aborting...\n\n",
              output->filename);
        exit(1);
      }
    }
  }

  if ((simdata_rank->pold = allocate_data_rank(&sim_grid)) == NULL ||
      (simdata_rank->pnew = allocate_data_rank(&sim_grid)) == NULL ||
      (simdata_rank->vxold = allocate_data_rank(&sim_grid)) == NULL ||
      (simdata_rank->vxnew = allocate_data_rank(&sim_grid)) == NULL ||
      (simdata_rank->vyold = allocate_data_rank(&sim_grid)) == NULL ||
      (simdata_rank->vynew = allocate_data_rank(&sim_grid)) == NULL ||
      (simdata_rank->vzold = allocate_data_rank(&sim_grid)) == NULL ||
      (simdata_rank->vznew = allocate_data_rank(&sim_grid)) == NULL) {
    printf("Failed to allocate memory. Aborting...\n\n");
    exit(1);
  }


  fill_data_rank(simdata_rank,simdata_rank->pold, 0.0);
  fill_data_rank(simdata_rank,simdata_rank->pnew, 0.0);

  fill_data_rank(simdata_rank,simdata_rank->vynew, 0.0);
  fill_data_rank(simdata_rank,simdata_rank->vxold, 0.0);
  fill_data_rank(simdata_rank,simdata_rank->vynew, 0.0);
  fill_data_rank(simdata_rank,simdata_rank->vyold, 0.0);
  fill_data_rank(simdata_rank,simdata_rank->vznew, 0.0);
  fill_data_rank(simdata_rank,simdata_rank->vzold, 0.0);
  
  printf("\n");
  printf(" Grid spacing: %g\n", simdata_rank->params.dx);
  printf("  Grid size X: %d\n", sim_grid.numnodesx);
  printf("  Grid size Y: %d\n", sim_grid.numnodesy);
  printf("  Grid size Z: %d\n", sim_grid.numnodesz);
  printf("    Time step: %g\n", simdata_rank->params.dt);
  printf(" Maximum time: %g\n\n", simdata_rank->params.maxt);

  if (simdata_rank->params.outrate > 0 && simdata_rank->params.outputs) {
    int outsampling =
        (int)(1.0 / (simdata_rank->params.outrate * simdata_rank->params.dt));

    printf("     Output rate: every %d step(s)\n", simdata_rank->params.outrate);
    printf(" Output sampling: %d Hz\n\n", outsampling);
    printf(" Output files:\n\n");

    for (int i = 0; i < simdata_rank->params.numoutputs; i++) {
      print_output(&simdata_rank->params.outputs[i]);
    }

    printf("\n");

  } else if (simdata_rank->params.outrate < 0) {
    printf("  Output is disabled (output rate set to 0)\n\n");

  } else {
    printf("  Output is disabled (not output specified)\n\n");
  }

  print_source(&simdata_rank->params.source);

  fflush(stdout);

  free(rho_map->vals);
  free(rho_map);
  free(c_map->vals);
  free(c_map);
  //DEBUG_PRINT("End of init simulation");
    
  }
    

    void finalize_simulation(simulation_data_rank_t *simdata_rank) {
      if (simdata_rank->params.outputs != NULL) {
        for (int i = 0; i < simdata_rank->params.numoutputs; i++) {
          free(simdata_rank->params.outputs[i].filename);

          if (simdata_rank->params.outrate > 0) {
            fclose(simdata_rank->params.outputs[i].fp);
          }
        }

        free(simdata_rank->params.outputs);
      }

      free(simdata_rank->params.source.data);
      free(simdata_rank->params.cin_filename);
      free(simdata_rank->params.rhoin_filename);

      free(simdata_rank->rho->vals);
      free(simdata_rank->rho);
      free(simdata_rank->rhohalf->vals);
      free(simdata_rank->rhohalf);
      free(simdata_rank->c->vals);
      free(simdata_rank->c);

      free(simdata_rank->pold->vals);
      free(simdata_rank->pold);
      free(simdata_rank->pnew->vals);
      free(simdata_rank->pnew);

      free(simdata_rank->vxold->vals);
      free(simdata_rank->vxold);
      free(simdata_rank->vxnew->vals);
      free(simdata_rank->vxnew);
      free(simdata_rank->vyold->vals);
      free(simdata_rank->vyold);
      free(simdata_rank->vynew->vals);
      free(simdata_rank->vynew);
      free(simdata_rank->vzold->vals);
      free(simdata_rank->vzold);
      free(simdata_rank->vznew->vals);
      free(simdata_rank->vznew);
  }
  


void swap_timesteps(simulation_data_rank_t *simdata) {
  data_rank_t *tmpp = simdata->pold;
  data_rank_t *tmpvx = simdata->vxold;
  data_rank_t *tmpvy = simdata->vyold;
  data_rank_t *tmpvz = simdata->vzold;

  simdata->pold = simdata->pnew;
  simdata->pnew = tmpp;
  simdata->vxold = simdata->vxnew;
  simdata->vxnew = tmpvx;
  simdata->vyold = simdata->vynew;
  simdata->vynew = tmpvy;
  simdata->vzold = simdata->vznew;
  simdata->vznew = tmpvz;
}

