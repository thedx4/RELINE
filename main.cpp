#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <boost/thread.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
// Sun CC doesn't handle boost::iterator_adaptor yet
#if !defined(__SUNPRO_CC) || (__SUNPRO_CC > 0x530)
#include <boost/generator_iterator.hpp>
#endif


#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

std::map<std::string, int> hashmap_l;
std::map<std::string, int> hashmap_r;
std::map<std::string, int> hashmap_t;
std::map<std::string, int> hashmap_l_u;
std::map<std::string, int> hashmap_st;
char poi_file[MAX_STRING], net_poi[MAX_STRING], net_poi_reg[MAX_STRING], net_poi_time[MAX_STRING], net_poi_word[MAX_STRING];
char emb_poi[MAX_STRING], emb_reg[MAX_STRING], emb_time[MAX_STRING], emb_word[MAX_STRING];
char poi_file_u[MAX_STRING], net_poi_u[MAX_STRING], net_poi_reg_u[MAX_STRING], net_poi_time_u[MAX_STRING], net_poi_word_u[MAX_STRING], net_poi_st[MAX_STRING];
char emb_poi_u[MAX_STRING], emb_reg_u[MAX_STRING], emb_time_u[MAX_STRING], emb_word_u[MAX_STRING], emb_st[MAX_STRING];
std::vector <std::string> vertex_poi_name;
std::vector <double> vertex_poi_degree;
std::vector <std::string> vertex_v_name;
std::vector <double> vertex_v_degree;
std::vector <std::string> vertex_r_name;
std::vector <double> vertex_r_degree;
std::vector <std::string> vertex_t_name;
std::vector <double> vertex_t_degree;
std::vector <std::string> vertex_poiu_name;
std::vector <double> vertex_poiu_degree;
std::vector <std::string> vertex_vu_name;
std::vector <double> vertex_vu_degree;
std::vector <std::string> vertex_ru_name;
std::vector <double> vertex_ru_degree;
std::vector <std::string> vertex_tu_name;
std::vector <double> vertex_tu_degree;
std::vector <std::string> vertex_st_name;
std::vector <double> vertex_st_degree;

int is_binary = 0, num_threads = 10, dim = 20, num_negative = 5;
char path [MAX_STRING]="";

int *neg_table_v, *neg_table_r, *neg_table_t;
//USER
int *neg_table_vu;
//STAYPOINT
int *neg_table_st;

int num_vertices_poi = 0, num_vertices_v = 0, num_vertices_r = 0, num_vertices_t = 0;
long long total_samples = 100, current_sample_count = 0, num_edges_vv = 0, num_edges_vr = 0, num_edges_vt = 0, num_edges_vw = 0;
//USER
int num_vertices_poiu = 0, num_vertices_vu = 0;
long long num_edges_vvu = 0, num_edges_vru = 0, num_edges_vtu = 0, num_edges_vwu = 0;
//STAYPOINT
int num_vertices_st = 0;
long long num_edges_st = 0;

real init_rho = 0.025, rho;
real **emb_vertex_v, **emb_vertex_r, **emb_vertex_t, *sigmoid_table;
//USER
real **emb_vertex_vu;
//STAYPOINT
real **emb_vertex_st;

int *vv_edge_source_id, *vv_edge_target_id, *vr_edge_source_id, *vr_edge_target_id, *vt_edge_source_id, *vt_edge_target_id, *vw_edge_source_id, *vw_edge_target_id;
double *vv_edge_weight, *vr_edge_weight, *vt_edge_weight, *vw_edge_weight;
//USER
int *vvu_edge_source_id, *vvu_edge_target_id, *vru_edge_source_id, *vru_edge_target_id, *vtu_edge_source_id, *vtu_edge_target_id, *vwu_edge_source_id, *vwu_edge_target_id;
double *vvu_edge_weight, *vru_edge_weight, *vtu_edge_weight, *vwu_edge_weight;
//USER
int *st_edge_source_id, *st_edge_target_id;
double *st_edge_weight;

// Parameters for edge sampling
long long *alias_vv, *alias_vr, *alias_vt, *alias_vw;
double *prob_vv, *prob_vr, *prob_vt, *prob_vw;
//USER
// Parameters for edge sampling USER
long long *alias_vvu, *alias_vru, *alias_vtu, *alias_vwu;
double *prob_vvu, *prob_vru, *prob_vtu, *prob_vwu;
// Parameters for edge sampling Stay Point
long long *alias_st;
double *prob_st;

//random generator 
typedef boost::minstd_rand base_generator_type;
base_generator_type generator(42u);
boost::uniform_real<> uni_dist(0, 1);
boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);

/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key)
{
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key)
	{
		hash = hash * seed + (*key++);
	}
	//return hash % hash_table_size; // LOL!
	return -1;
}

void InsertHashTable(char *key, int value, int flag)
{
	std::string str(key);
	if (flag==0) {
		hashmap_l[str]=value;
	}else if (flag==1){
		hashmap_r[str]=value;
	}else if (flag==2){
		hashmap_t[str]=value;
	}else if (flag==3){
		hashmap_l_u[str]=value;
	}else if (flag==4){
		hashmap_l_u[str]=value;
	}else if (flag==5){
		hashmap_r[str]=value;
	}else if (flag==6){
		hashmap_t[str]=value;
	}else if (flag==7){
		hashmap_l[str]=value;
	}else if (flag==8){
		hashmap_st[str]=value;
	}else{
		std::cout<<"Error on flag"<<std::endl;
		exit(1);
	}
}

//int SearchHashTable(char *key, ClassVertex *vertex, int flag)
int SearchHashTable(char *key, int flag)
{
	std::string str(key);
	//std::cout<<"Entering search"<<std::endl;
	if (flag==0) {
		if(!(hashmap_l.find(str) != hashmap_l.end())){
			return -1;
		}else{
			return hashmap_l[str];
		}
	}else if (flag==1){
		if(!(hashmap_r.find(str) != hashmap_r.end())){
			return -1;
		}else{
			return hashmap_r[str];
		}
	}else if (flag==2){
		if(!(hashmap_t.find(str) != hashmap_t.end())){
			return -1;
		}else{
			return hashmap_t[str];
		}
	}else if (flag==3){
		if(!(hashmap_l_u.find(str) != hashmap_l_u.end())){
			return -1;
		}else{
			return hashmap_l_u[str];
		}
	}else if (flag==4){
		if(!(hashmap_l_u.find(str) != hashmap_l_u.end())){
			return -1;
		}else{
			return hashmap_l_u[str];
		}
	}else if (flag==5){
		if(!(hashmap_r.find(str) != hashmap_r.end())){
			return -1;
		}else{
			return hashmap_r[str];
		}
	}else if (flag==6){
		if(!(hashmap_t.find(str) != hashmap_t.end())){
			return -1;
		}else{
			return hashmap_t[str];
		}
	}else if (flag==7){
		if(!(hashmap_l.find(str) != hashmap_l.end())){
			return -1;
		}else{
			return hashmap_l[str];
		}
	}else if (flag==8){
		if(!(hashmap_st.find(str) != hashmap_st.end())){
			return -1;
		}else{
			return hashmap_st[str];
		}
	}else{
		std::cout<<"Error on flag"<<std::endl;
		exit(1);
	}
	
	std::cout<<"Error on search"<<std::endl;
	exit(1);
}

/* Add a vertex to the vertex set */
int AddVertex(char *name, std::vector<std::string> &vertex, std::vector<double> &vertexd, int &num_vertices, int flag)
{
	std::string prox (name);
	vertex.push_back(name);
	vertexd.push_back(0);
	num_vertices++;
	InsertHashTable(name, num_vertices - 1, flag);
	return num_vertices - 1;
}


/* Read network from the training file */
void ReadFile(char *network_file, long long &num_edges, int &num_vertices, 
	          int *&edge_source_id, int *&edge_target_id, double *&edge_weight, std::vector <std::string> &vertex, std::vector <double> &vertexd, int hash_flag, int flag, int dxflag)
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid;
	double weight;

	fin = fopen(network_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}
	num_edges = 0;
	while (fgets(str, sizeof(str), fin)) num_edges++;
	fclose(fin);
	//printf("Number of edges: %lld          \n", num_edges);
	
	edge_source_id = new int[num_edges];
	edge_target_id = new int[num_edges];
	edge_weight = new double[num_edges];
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	fin = fopen(network_file, "rb");

	//if( (dxflag==2) | (hash_flag==4) | (hash_flag!=3&&dxflag==0) )num_vertices = 0;
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		//if(hash_flag==5)std::cout<<"Now at line: "<<k<<std::endl;
		if (flag==1){
			if(dxflag==0)vid = SearchHashTable(name_v1, 0);
			if(dxflag==1)vid = SearchHashTable(name_v1, 4);
			if(dxflag==2)vid = SearchHashTable(name_v1, 1);
			if (vid == -1) std::cout<<"Error: false point type on line "<<k+1<<" : "<<name_v1<<" flag="<<flag<<std::endl;
			if (vertexd[vid] == 0) {num_vertices++;}
			vertexd[vid] += weight;
			edge_source_id[k] = vid;
		}
		else{
			if(dxflag==0)vid = SearchHashTable(name_v1, 0);
			if(dxflag==1)vid = SearchHashTable(name_v1, 4);
			if(dxflag==2)vid = SearchHashTable(name_v1, 1);
			edge_source_id[k] = vid;
		}
		//if(hash_flag==5)std::cout<<"1"<<std::endl;
 		vid = SearchHashTable(name_v2, hash_flag);
		if (vid == -1) vid = AddVertex(name_v2, vertex, vertexd, num_vertices, hash_flag);
		if (flag == 1 && vertexd[vid] == 0) {num_vertices++;}
		//if(hash_flag==5)std::cout<<"4 vid = "<<vid<<std::endl;
		vertexd[vid] += weight;
		//if(hash_flag==5)std::cout<<"5"<<std::endl;
		edge_target_id[k] = vid;
		//if(hash_flag==5)std::cout<<"6"<<std::endl;
		
		edge_weight[k] = weight;
	}
	fclose(fin);
	//printf("Number of vertices: %lld          \n", num_vertices);
}

void ReadPOIs(char *POI_file){
	FILE *fin;
	char name[MAX_STRING], str[MAX_STRING+10];
	int num_poi = 0, vid;

	fin = fopen(POI_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}

	while (fgets(str, sizeof(str), fin)) num_poi++;
	fclose(fin);

	fin = fopen(POI_file, "rb");
	for (int k = 0; k != num_poi; k++)
	{
		fscanf(fin, "%s", name);
		vid = SearchHashTable(name, 0);
		if (vid == -1) {
			vid = AddVertex(name, vertex_poi_name, vertex_poi_degree, num_vertices_poi, 0);
		}
	}
	fclose(fin);
}

void ReadUSERs(char *USERS_file){
	FILE *fin;
	char name[MAX_STRING], str[MAX_STRING+10];
	int num_user = 0, vid;

	fin = fopen(USERS_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}

	while (fgets(str, sizeof(str), fin)) num_user++;
	fclose(fin);

	fin = fopen(USERS_file, "rb");
	for (int k = 0; k != num_user; k++)
	{
		fscanf(fin, "%s", name);
		vid = SearchHashTable(name, 4);
		if (vid == -1) {
			vid = AddVertex(name, vertex_poiu_name, vertex_poiu_degree, num_vertices_poiu, 4);
		}
	}
	fclose(fin);
}

void ReadData(){
	//char *name;
	std::cout<<std::endl<<"Locations"<<std::endl;
	//int max_num = 1000;
	/* Init vertex_v* 's v poit in different graph */
	for(int i=0; i<num_vertices_poi; i++){
		vertex_v_name.push_back(vertex_poi_name[i]);
		vertex_v_degree.push_back(vertex_poi_degree[i]);
	}

	
	ReadFile(net_poi, num_edges_vv, num_vertices_v, vv_edge_source_id, vv_edge_target_id, vv_edge_weight, vertex_v_name, vertex_v_degree, 0, 1, 0);
	std::cout<<"Graph Loc-Loc: "<<"\t"<<"\t"<<num_vertices_v<<"x"<<num_vertices_v;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vv<<"\n";
	
	std::cout<<std::endl<<"Users"<<std::endl;
	
	//max_num = 1000;
	/* Init vertex_vu* 's v poit in different graph */
	for(int i=0; i<num_vertices_poiu; i++){
		vertex_vu_name.push_back(vertex_poiu_name[i]);
		vertex_vu_degree.push_back(vertex_poiu_degree[i]);
	}
	
	ReadFile(net_poi_u, num_edges_vvu, num_vertices_vu, vvu_edge_source_id, vvu_edge_target_id, vvu_edge_weight, vertex_vu_name, vertex_vu_degree, 4, 1, 1);
	std::cout<<"Graph User-User: "<<"\t"<<num_vertices_vu<<"x"<<num_vertices_vu;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vvu<<"\n";
	
	std::cout<<std::endl<<"Location Networks"<<std::endl;
	
	//max_num_vertices = 1000;
	for(int i=0; i<num_vertices_poiu; i++){
		vertex_vu_name.push_back(vertex_poiu_name[i]);
		vertex_vu_degree.push_back(vertex_poiu_degree[i]);
	}
	ReadFile(net_poi_reg, num_edges_vr, num_vertices_r, vr_edge_source_id, vr_edge_target_id, vr_edge_weight, vertex_r_name, vertex_r_degree, 1, 0, 0);
	std::cout<<"Graph Loc-Route: "<<"\t"<<num_vertices_v<<"x"<<num_vertices_r;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vr<<"\n";

	//max_num_vertices = 1000;
	ReadFile(net_poi_time, num_edges_vt, num_vertices_t, vt_edge_source_id, vt_edge_target_id, vt_edge_weight, vertex_t_name, vertex_t_degree, 2, 0, 0);
	std::cout<<"Graph Loc-Time: "<<"\t"<<num_vertices_v<<"x"<<num_vertices_t;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vt<<"\n";

	//max_num_vertices = 1000;
	ReadFile(net_poi_word, num_edges_vw, num_vertices_vu, vw_edge_source_id, vw_edge_target_id, vw_edge_weight, vertex_vu_name, vertex_vu_degree, 3, 0, 0);
	std::cout<<"Graph Loc-User: "<<"\t"<<num_vertices_v<<"x"<<num_vertices_vu;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vw<<"\n";
	
	//USER
	std::cout<<std::endl<<"User Networks"<<std::endl;
	
	//max_num_vertices = 1000;
	for(unsigned int i=0; i<vertex_r_name.size(); i++){
		vertex_ru_name.push_back(vertex_r_name[i]);
		vertex_ru_degree.push_back(0);
	}
	ReadFile(net_poi_reg_u, num_edges_vru, num_vertices_r, vru_edge_source_id, vru_edge_target_id, vru_edge_weight, vertex_ru_name, vertex_ru_degree, 5, 0, 1);
	std::cout<<"Graph User-Route: "<<"\t"<<num_vertices_vu<<"x"<<num_vertices_r;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vru<<"\n";

	//max_num_vertices = 1000;
	for(unsigned int i=0; i<vertex_t_name.size(); i++){
		vertex_tu_name.push_back(vertex_t_name[i]);
		vertex_tu_degree.push_back(0);
	}
	ReadFile(net_poi_time_u, num_edges_vtu, num_vertices_t, vtu_edge_source_id, vtu_edge_target_id, vtu_edge_weight, vertex_tu_name, vertex_tu_degree, 6, 0, 1);
	std::cout<<"Graph User-Time: "<<"\t"<<num_vertices_vu<<"x"<<num_vertices_t;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vtu<<"\n";

	//max_num_vertices = 1000;
	ReadFile(net_poi_word_u, num_edges_vwu, num_vertices_v, vwu_edge_source_id, vwu_edge_target_id, vwu_edge_weight, vertex_v_name, vertex_v_degree, 7, 0, 1);
	std::cout<<"Graph User-Loc: "<<"\t"<<num_vertices_vu<<"x"<<num_vertices_v;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_vwu<<"\n";
	
	//STAYPOINTS
	ReadFile(net_poi_st, num_edges_st, num_vertices_st, st_edge_source_id, st_edge_target_id, st_edge_weight, vertex_st_name, vertex_st_degree, 8, 0, 2);
	std::cout<<"Graph Route-StayPoint: "<<"\t"<<num_vertices_r<<"x"<<num_vertices_st;
	std::cout<<"\t"<<"Number of edges:"<<"\t";
	std::cout<<num_edges_st<<"\n";
}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable(long long *&alias, double *&prob, long long num_edges, double *edge_weight)
{
	//alias = (long long *)malloc(num_edges*sizeof(long long));
	//prob = (double *)malloc(num_edges*sizeof(double));
	alias = new long long[num_edges];
	prob = new double [num_edges];
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	//double *norm_prob = (double*)mallocmalloc(num_edges*sizeof(double));
	//long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	//long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
	
	double *norm_prob = new double [num_edges];
	long long  *large_block = new long long [num_edges];
	long long  *small_block = new long long [num_edges];
	
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
	for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

	for (long long k = num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block];
		alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;
}

void InitAlias()
{
	InitAliasTable(alias_vv, prob_vv, num_edges_vv, vv_edge_weight);
	InitAliasTable(alias_vr, prob_vr, num_edges_vr, vr_edge_weight);
	InitAliasTable(alias_vt, prob_vt, num_edges_vt, vt_edge_weight);
	InitAliasTable(alias_vw, prob_vw, num_edges_vw, vw_edge_weight);
	
	//USER
	
	InitAliasTable(alias_vvu, prob_vvu, num_edges_vvu, vvu_edge_weight);
	InitAliasTable(alias_vru, prob_vru, num_edges_vru, vru_edge_weight);
	InitAliasTable(alias_vtu, prob_vtu, num_edges_vtu, vtu_edge_weight);
	InitAliasTable(alias_vwu, prob_vwu, num_edges_vwu, vwu_edge_weight);
	
	//ST
	
	InitAliasTable(alias_st, prob_st, num_edges_st, st_edge_weight);
}

long long SampleAnEdge(double rand_value1, double rand_value2, int num_edges, long long *alias, double *prob)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
	long long a, b;
	//int num;
	//vertex of poi
	emb_vertex_v = new real *[num_vertices_poi];
	for (int dx=0; dx<num_vertices_poi;dx++) emb_vertex_v[dx] = new real [dim];
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_poi; a++)
		emb_vertex_v[a][b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_v[a * dim + b] = 0;
	
	//vertex of region
	emb_vertex_r = new real *[num_vertices_r];
	for (int dx=0; dx<num_vertices_r;dx++) emb_vertex_r[dx] = new real [dim];
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_r; a++)
		emb_vertex_r[a][b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_r[a * dim + b] = 0;
	
	//vertex of time
	emb_vertex_t = new real *[num_vertices_t];
	for (int dx=0; dx<num_vertices_t;dx++) emb_vertex_t[dx] = new real [dim];
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_t; a++)
		emb_vertex_t[a][b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_t[a * dim + b] = 0;
	
	//USER
	emb_vertex_vu = new real *[num_vertices_poiu];
	for (int dx=0; dx<num_vertices_poiu;dx++) emb_vertex_vu[dx] = new real [dim];
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_poiu; a++)
		emb_vertex_vu[a][b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_vu[a * dim + b] = 0;
		
	//Stay Point
	emb_vertex_st = new real *[num_vertices_st];
	for (int dx=0; dx<num_vertices_st;dx++) emb_vertex_st[dx] = new real [dim];
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_st; a++)
		emb_vertex_st[a][b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_st[a * dim + b] = 0;
		
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable(int *&neg_table, int num_vertices, std::vector <double> &vertexd)
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	//neg_table = (int *)malloc(neg_table_size * sizeof(int));
	neg_table = new int[neg_table_size];
	
	for (int k = 0; k != num_vertices; k++) sum += pow(vertexd[k], NEG_SAMPLING_POWER);
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertexd[vid], NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}

void InitNeg()
{
	InitNegTable(neg_table_v, num_vertices_v, vertex_v_degree);
	InitNegTable(neg_table_r, num_vertices_r, vertex_r_degree);
	InitNegTable(neg_table_t, num_vertices_t, vertex_t_degree);
	
	//USER
	
	InitNegTable(neg_table_vu, num_vertices_vu, vertex_vu_degree);
	
	//Stay Point
	
	InitNegTable(neg_table_st, num_vertices_st, vertex_st_degree);
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
	real x;
	//sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	sigmoid_table = new real[sigmoid_table_size + 1];
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainLINEThread(void *id)
{
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	int *neg_table, *edge_source_id, *edge_target_id;
	real **emb_vertex_target;
	//real *vec_error = (real *)calloc(dim, sizeof(real));
	//real *vec_erroru = (real *)calloc(dim, sizeof(real));
	real *vec_error =  new real [dim];
	real *vec_erroru =  new real [dim];
	real *vec_errorst =  new real [dim];
	
	while (1)
	{
		if (count > total_samples / num_threads + 2) break;
		
		if (count - last_count>10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			//printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		int a = count%9;
		
		switch(a){
		case 0:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vr, alias_vr, prob_vr);
			neg_table = neg_table_r;
			emb_vertex_target = emb_vertex_r;
			edge_source_id = vr_edge_source_id;
			edge_target_id = vr_edge_target_id;
			break;
		case 1:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vt, alias_vt, prob_vt);
			neg_table = neg_table_t;
			emb_vertex_target = emb_vertex_t;
			edge_source_id = vt_edge_source_id;
			edge_target_id = vt_edge_target_id;
			break;
		case 2:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vw, alias_vw, prob_vw);
			neg_table = neg_table_vu;
			emb_vertex_target = emb_vertex_vu;
			edge_source_id = vw_edge_source_id;
			edge_target_id = vw_edge_target_id;
			break;
		case 3:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vv, alias_vv, prob_vv);
			neg_table = neg_table_v;
			emb_vertex_target = emb_vertex_v;
			edge_source_id = vv_edge_source_id;
			edge_target_id = vv_edge_target_id;
			break;
		case 4:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vru, alias_vru, prob_vru);
			neg_table = neg_table_r;
			emb_vertex_target = emb_vertex_r;
			edge_source_id = vru_edge_source_id;
			edge_target_id = vru_edge_target_id;
			break;
		case 5:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vtu, alias_vtu, prob_vtu);
			neg_table = neg_table_t;
			emb_vertex_target = emb_vertex_t;
			edge_source_id = vtu_edge_source_id;
			edge_target_id = vtu_edge_target_id;
			break;
		case 6:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vwu, alias_vwu, prob_vwu);
			neg_table = neg_table_v;
			emb_vertex_target = emb_vertex_v;
			edge_source_id = vwu_edge_source_id;
			edge_target_id = vwu_edge_target_id;
			break;
		case 7:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vvu, alias_vvu, prob_vvu);
			neg_table = neg_table_vu;
			emb_vertex_target = emb_vertex_vu;
			edge_source_id = vvu_edge_source_id;
			edge_target_id = vvu_edge_target_id;
			break;
		case 8:
			curedge = SampleAnEdge(uni(), uni(), num_edges_st, alias_st, prob_st);
			neg_table = neg_table_st;
			emb_vertex_target = emb_vertex_st;
			edge_source_id = st_edge_source_id;
			edge_target_id = st_edge_target_id;
			break;
		default:
			std::cout<<"Error in Case"<<std::endl;
			exit(1);
		}

		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

		
		lu = u;
		if(a<=3)for (int c = 0; c != dim; c++) vec_error[c] = 0;
		if(a>3&&a<8)for (int c = 0; c != dim; c++) vec_erroru[c] = 0;
		if(a==8)for (int c = 0; c != dim; c++) vec_errorst[c] = 0;

		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			if (d == 0)
			{
				target = v;
				label = 1;
			}
			else
			{
				target = neg_table[Rand(seed)];
				label = 0;
			}
			lv = target;
			if(a<=3)Update(emb_vertex_v[lu], emb_vertex_target[lv], vec_error, label);
			if(a>3&&a<8)Update(emb_vertex_vu[lu], emb_vertex_target[lv], vec_erroru, label);
			if(a==8)Update(emb_vertex_r[lu], emb_vertex_target[lv], vec_errorst, label);
		}
		if(a<=3)for (int c = 0; c != dim; c++) emb_vertex_v[lu][c] += vec_error[c];
		if(a>3&&a<8)for (int c = 0; c != dim; c++) emb_vertex_vu[lu][c] += vec_erroru[c];
		if(a==8)for (int c = 0; c != dim; c++) emb_vertex_r[lu][c] += vec_errorst[c];
		count++;
	}
	return NULL;
}

void OutputFile(char emb_file[100], int num_vertices, std::vector <std::string> &vertex, real **emb_vertex){
	std::cout<<"outputfile... "<<num_vertices<<"\n";
	
	std::ofstream outputfile;
	outputfile.open (emb_file,std::ios::trunc);

	outputfile << num_vertices <<" "<<dim<<"\n";
	for (int a = 0; a < num_vertices; a++)
	{
		outputfile << vertex[a].c_str();
		for (int b = 0; b < dim; b++) outputfile << " " << emb_vertex[a][b];
		outputfile<< "\n";
	}
	outputfile.close();
}

void Output()
{
	OutputFile(emb_poi, num_vertices_poi, vertex_v_name, emb_vertex_v);
	OutputFile(emb_reg, num_vertices_r, vertex_r_name, emb_vertex_r);
	OutputFile(emb_time, num_vertices_t, vertex_t_name, emb_vertex_t);
	OutputFile(emb_word, num_vertices_vu, vertex_vu_name, emb_vertex_vu);
	
	//USER
	
	//OutputFile(emb_poi_u, num_vertices_poiu, vertex_vu_name, emb_vertex_vu);
	//OutputFile(emb_reg_u, num_vertices_r, vertex_ru_name, emb_vertex_r);
	//OutputFile(emb_time_u, num_vertices_t, vertex_tu_name, emb_vertex_t);
	//OutputFile(emb_word_u, num_vertices_v, vertex_v_name, emb_vertex_v);
	
	//STAYPOINTS
	OutputFile(emb_st, num_vertices_st, vertex_st_name, emb_vertex_st);
}

void TrainLINE() {
	long a;
	boost::thread *pt = new boost::thread[num_threads];

	printf("--------------------------------\n");
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("Thread: %d\n", num_threads);
	printf("--------------------------------\n");
	
	std::cout<<"Reading Location File..."<<std::flush;
	ReadPOIs(poi_file);
	std::cout<<"\tSuccess"<<std::endl<<"Reading Users File..."<<std::flush;
	ReadUSERs(poi_file_u);
	std::cout<<"\tSuccess"<<std::endl<<"Reading Data..."<<std::flush;
	ReadData();
	std::cout<<"Success"<<std::endl<<"Initializing Alias..."<<std::flush;
	InitAlias();
	std::cout<<"\tSuccess"<<std::endl<<"Initializing Vectors..."<<std::flush;
	InitVector();
	std::cout<<"\tSuccess"<<std::endl<<"Initializing Negative tables..."<<std::flush;
	InitNeg();
	std::cout<<"\tSuccess"<<std::endl<<"Initializing Sigmoid table..."<<std::flush;
	InitSigmoidTable();
	
	std::cout<<"\tSuccess!!"<<std::endl;
	
	
	clock_t start = clock();
	printf("--------------------------------\n");
	for (a = 0; a < num_threads; a++) pt[a] = boost::thread(TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pt[a].join();
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	Output();
	std::cout<<"JUPDGE finish..... "<<"\n";
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("LINE: Large Information Network Embedding\n\n");
		printf("Options:\n");
		printf("\t-path <int>\n");
		printf("\t\tPath to files\n");
		printf("Parameters for training:\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 100\n");
		printf("\t-order <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\nExamples:\n");
		printf("./JNGEmodel -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
		return 0;
	}

	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-path", argc, argv)) > 0) strcpy (path, argv[i + 1]);
	
	strcpy (poi_file, path);
	strncat(poi_file, "pois.txt",10);

	strcpy (net_poi, path);
	strcpy (net_poi_word, path);
	strcpy (net_poi_time, path);
	strcpy (net_poi_reg, path);
	strncat(net_poi, "ll.txt",10); 
	strncat(net_poi_word, "lu.txt",10); 
	strncat(net_poi_time, "lt.txt",10); 
	strncat(net_poi_reg, "lr.txt",10); 
	
	strcpy (emb_poi, path);
	strcpy (emb_reg, path);
	strcpy (emb_time, path);
	strcpy (emb_word, path);
	strncat(emb_poi, "ll_v.txt",10);
	strncat(emb_reg, "lr_v.txt",10);
	strncat(emb_time, "lt_v.txt",10);
	strncat(emb_word, "lu_v.txt",10);
	
	strcpy (poi_file_u, path);
	strncat(poi_file_u, "users.txt",10);

	strcpy (net_poi_u, path);
	strcpy (net_poi_word_u, path);
	strcpy (net_poi_time_u, path);
	strcpy (net_poi_reg_u, path);
	strncat(net_poi_u, "uu.txt",10); 
	strncat(net_poi_word_u, "ul.txt",10); 
	strncat(net_poi_time_u, "ut.txt",10); 
	strncat(net_poi_reg_u, "ur.txt",10); 
	
	strcpy (emb_poi_u, path);
	strcpy (emb_reg_u, path);
	strcpy (emb_time_u, path);
	strcpy (emb_word_u, path);
	strncat(emb_poi_u, "uu_v.txt",10);
	strncat(emb_reg_u, "ur_v.txt",10);
	strncat(emb_time_u, "ut_v.txt",10);
	strncat(emb_word_u, "ul_v.txt",10);
	
	strcpy (net_poi_st, path);
	strncat(net_poi_st, "rsp.txt",10); 
	
	strcpy (emb_st, path);
	strncat(emb_st, "rsp_v.txt",10);
	
	total_samples *= 1000000;
	rho = init_rho;
	
	TrainLINE();
	return 0;
}
