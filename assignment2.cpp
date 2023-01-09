//slurm file:(just a small demo, you can modify city number to test bigger instance if you wish)
/*
#!/bin/bash
#SBATCH --partition=snowy
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:5:00
#SBATCH --mem=4G
module load gcc
module load gompi
mpicxx -o yuhuo yuzhouhuo_assignment2.cpp -fopenmp -O3 
mpiexec -n 8 yuhuo
*/

/////////
// referred code:https://zhuanlan.zhihu.com/p/45985636
// This program can theoretically read from input files, 
// (by modifying the random parameter in matgen func call from main )
// however, as the submission requires single C++ file
// the graph matrix used to presentation this program is randomly generated with random seed=0
/////////
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<memory.h>
#include<omp.h>
#include<mpi.h>
#include<time.h>
#ifdef __linux
#include<sys/time.h>
#endif // _linux
#include<random>
using namespace std;
//about ants
#define USEINPUT false
//if USEINPUT is false, this program randomly generate graph matrix with random seed 0
//otherwise, it read from input(compiled on linux) or from localPath (compiled on windows)
//the input contains n+1 lines, the first line is city number n
//the remaining lines are [cityId cityXloc cityYloc] ints
#define ALPHA 1 //weight of historical info
#define BETA 5  //weight of length info
#define P 0.7   //rate of history info loss
#define Q 100  //the lnfo per iter
#define MIN_PHER_RATIO 1e-4//min concentration ratio of pheromones 
#define MAX_PHER_RATIO 10//max concentration ratio of pheromones
#define EPS 1e-8 //a small number
#define INF 1000000//a big number
#define MAX_ITER 100//total iter num of nests
#define ANT_ITER 5//the loop number in a nest loop
#define FOREIGN_WEIGHT 1//the weight of info from outer nest
#define MAX_RESTART_ITER 100
//runtime settings
#define ANT_NUM_RATIO 0.8 //the ratio of ant number/city number
#define MAX_OPT2_TIMES 2//if max opt2 times>1, it will restart and do at most MAX_OPT2_TIMES-1 opt2
//problem generation settings
#define CITY_NUMB 600//city number
#define ROAD_DENSITY 1.0//density of road between cities
#define MAX_ROAD_LEN 100//max road length
#define FULL_PATH_LEN (cityNumbers+3)
#define MAX_THREAD_NUMB 6//max thread number taken into account by load balancer
#define INDIVIDUAL_UPDATE TRUE
//#define CUR_RND rnd_fixed
//global variables
int cityNumbers;
int rank = 0;
int threadNum;

//global variable generated from runtime
int myid;
int numprocs;
int maxIter; //the number of iter rounds of each nest
int antIter;
int foreignWeight;//the number of foreigh paths update executed
int restartInteval;//if the best hasn't updated for restartInteval, restart
double maxPher;
double minPher;
int updateList[3]{ 1,1,1 };
const double** curInfo;
int antNumber;
int** cityMap; //graph matrix
double** pheromone; //pheromones matrix
double** info;// info = pheromone ^ alpha * cityMap ^ beta
double** herustic;
string localPath = "E:\\ToSPC\\citys_small.txt";
double ompEfficiency[7]{ 0,0.0543,0.0917,0.1269,0.1400,0.1430,0.1547 };
int maxAntNumber;

#ifdef __linux
uint64_t GetTimeStamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
}
#endif
//----------------------------------utils--------------------------------------
int** request2Dint(int n) {
	int m = n + 1;
	int** dp = new int* [m];
	size_t size = m * m;
	int* dp0 = new int[size];
	if (!dp || !dp0)
	{
		std::cerr << "requesting 2D memory failed" << std::endl;
		exit(1);
	}
	dp[0] = dp0;

	for (int i = 1; i < m; i++) {
		dp[i] = dp[i - 1] + m;
	}
	return dp;
}

double** request2Ddouble(int n) {
	int m = n + 1;
	double** dp = new double* [m];
	size_t size = m * m;
	double* dp0 = new double[size];
	if (!dp || !dp0)
	{
		std::cerr << "requesting 2D memory failed" << std::endl;
		exit(1);
	}
	dp[0] = dp0;

	for (int i = 1; i < m; i++) {
		dp[i] = dp[i - 1] + m;
	}
	return dp;
}

inline void copyIntArr(int* source, int* dist, int n) {
	memcpy(dist, source, n * sizeof(int));
}

//efficient power algorithm copied from https://zhuanlan.zhihu.com/p/45985636
double power(double x, int y) {
	double ans = 1;
	while (y) {
		if (y & 1) ans *= x;
		x *= x;
		y >>= 1;
	}
	return ans;
}

//modified from https://zhuanlan.zhihu.com/p/45985636 
//for calculating distance
int euc_2d(int x1, int y1, int x2, int y2) {
	int tmp = (int)sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
	return tmp > 0 ? tmp : 1;
}

//used for print an array
void printPath(int* path, int n) {
	for (int i = 0; i < n; i++)
		printf("%d ", path[i]);
	printf("\n");
}

//----------------------------------mat related codes--------------------------------------
void matInit(int threadNum = 1) {
	int n = cityNumbers;
	cityMap = request2Dint(n);
	pheromone = request2Ddouble(n);
	herustic = request2Ddouble(n);
	info = request2Ddouble(n);
}

//init pheromone matrix
void pherInit(int threadNum = 1) {
	int n = cityNumbers;
	double sum = 0;
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
			sum += cityMap[i][j];
	sum = sum * 2 / (n * (n - 1));
	sum = 1 / sum;
	maxPher = MAX_PHER_RATIO * sum;//1;//
	minPher = MIN_PHER_RATIO * sum;//1e-4 ;//
	// intialising the info table
#pragma omp parallel for num_threads(threadNum) schedule(static) //proc_bind(close)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			if (i != j)
				pheromone[i][j] = maxPher;
			else
				pheromone[i][j] = minPher;
	}
}

//calculate herustic information (1/distance) for efficiency
void herusticInit(int threadNum = 1) {
	int n = cityNumbers;
#pragma omp parallel for num_threads(threadNum) schedule(static) //proc_bind(close)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			herustic[i][j] = 1 / (1.0 * cityMap[i][j] + EPS);
	}
}

//random generate a map as input from given seed
void matRandomGen(int seed, int threadNum = 1) {
	int n = cityNumbers;
	//srand((unsigned)time(NULL));
	srand(seed);
#pragma omp parallel for num_threads(threadNum) schedule(guided) //proc_bind(close)
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			if (rand() / double(RAND_MAX) < ROAD_DENSITY)
				cityMap[i][j] = (rand() % MAX_ROAD_LEN) + 1;
			else
				cityMap[i][j] = -1;
			cityMap[j][i] = cityMap[i][j];
		}
		cityMap[i][i] = 0;
	}
}

//generate a graph matrix as input, the input should contain n+1 rows
//the first row is the city number while the other is "city id x_loc y_loc"
void matLocRead(int threadNum = 1) {
	int* city_X = new int[cityNumbers + 1];
	int* city_Y = new int[cityNumbers + 1];
	int city, x, y;
	for (int i = 0; i < cityNumbers; i++) {
		cin >> city >> x >> y;
		city_X[city - 1] = x;
		city_Y[city - 1] = y;
	}
	for (int i = 0; i < cityNumbers; i++) {
		cityMap[i][i] = 0;
		for (int j = i + 1; j < cityNumbers; j++) {
			cityMap[i][j] = euc_2d(city_X[i], city_Y[i], city_X[j], city_Y[j]);
			cityMap[j][i] = cityMap[i][j];
		}
	}
	delete[] city_X;
	delete[] city_Y;
}

//can only be called by master
void matGen(bool random = true, int threadNum = 1) {
	if (random) {
		cityNumbers = CITY_NUMB;
		matInit();
		matRandomGen(0, threadNum);
	}
	else {
		FILE* fp = nullptr;
#ifdef  _WIN32
		//fopen_s(&fp,localPath.c_str(), "r");
		freopen_s(&fp, localPath.c_str(), "r", stdin);
#endif //  _WIN32
		cin >> cityNumbers;
		matInit();
		matLocRead(threadNum);
	}
}

void phermoreVolatile(int threadNum = 1) {
	int n = cityNumbers;
	double tmp;
#pragma omp parallel for num_threads(threadNum) schedule(static) //proc_bind(close)
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			tmp = pheromone[i][j] * P;
			pheromone[i][j] = tmp < minPher ? minPher : tmp;
		}
}

//append phermore to paths consisting given route
void phermoreMark(int* paths, int pathLen, int threadNum = 1) {
	int n = cityNumbers;
	double weight = 1.0 / pathLen;
#pragma omp parallel for if(n/threadNum>100) num_threads(threadNum) schedule(static) //proc_bind(close)
	for (int i = 0; i < n; i++) {
		pheromone[paths[i]][paths[i + 1]] += weight;
		pheromone[paths[i + 1]][paths[i]] += weight;
	}
}

//remove the phermore on the path consisting the worst route to accelerate coverge
void phermorePunish(int* paths, int threadNum = 1) {
	int n = cityNumbers;
	double tmp;
#pragma omp parallel for if(n/threadNum>100) num_threads(threadNum) schedule(static) //proc_bind(close)
	for (int i = 0; i < n; i++) {
		tmp = pheromone[paths[i]][paths[i + 1]] * P;
		tmp = tmp > minPher ? tmp : minPher;
		pheromone[paths[i]][paths[i + 1]] = tmp;
		pheromone[paths[i + 1]][paths[i]] = tmp;
	}
}

//the info matrix is strightforwardly the weight of cities
void updateInfoMatrix(int threadNum = 1) {
	int n = cityNumbers;
#pragma omp parallel for num_threads(threadNum) schedule(static) //proc_bind(close)
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			info[i][j] = power(pheromone[i][j], ALPHA) * power(herustic[i][j], BETA);
		}
}

void restart(int threadNum = 1) {
	herusticInit(threadNum);
	pherInit(threadNum);
	updateInfoMatrix(threadNum);
}


//----------------------------------ant codes--------------------------------------
class Ant {
protected:
	//simulate a set with arr
	int* unvisited;
	int cityPtr;
	//contain the paths generated by two ants
	int* forwardPath;
	int* backwardPath;
	//randomly select a city according to given weights
	int selectCity(int source, const double** curInfo = nullptr) {
		if (curInfo == nullptr)
			curInfo = (const double**)info;

		if (cityPtr == 0) {
			cityPtr--;
			return unvisited[0];
		}

		double sum_prob = 0, sum = 0;
		int tmp;
		for (int i = 0; i <= cityPtr; i++) {
			sum += curInfo[source][unvisited[i]];
		}
		double rnd = rand() / double(RAND_MAX) * sum;
		for (int i = 0; i <= cityPtr; i++) {
			sum_prob += curInfo[source][unvisited[i]];
			if (sum_prob >= rnd) {
				tmp = unvisited[i];
				unvisited[i] = unvisited[cityPtr--];
				return tmp;
			}
		}
		tmp = unvisited[cityPtr--];
		return tmp;

	}
	//reset ant state
	inline void init() {
		cityPtr = cityNumbers - 1;
		for (int i = 0; i < cityNumbers; i++)
			unvisited[i] = i;
		totalLen = INF;
		tag = -1;
	}
public:
	int* path; //For convinent the head node is repeated in tail
	int totalLen;
	int tag;
	//opt2 optimization, randomly swap two nodes in a route and save changes if generates better result
	void opt2(int* path, int times = 1) {
		int break1, break2, oldLen, newLen, tmp;
		int* swap = new int[cityNumbers + 1];
		for (int i = 0; i < times; i++) {
			break1 = rand() % cityNumbers + 1;
			break2 = rand() % cityNumbers + 1;
			if (break1 == break2)
				continue;
			if (break1 > break2) {
				tmp = break1;
				break1 = break2;
				break2 = tmp;
			}
			oldLen = cityMap[path[break1 - 1]][path[break1]] + cityMap[path[break2 - 1]][path[break2]];
			newLen = cityMap[path[break1 - 1]][path[break2 - 1]] + cityMap[path[break2]][path[break1]];
			if (newLen < oldLen) {
				break2--;
				while (break2 > break1) {
					tmp = path[break2];
					path[break2] = path[break1];
					path[break1] = tmp;
					break1++;
					break2--;
				}
				path[cityNumbers + 1] += newLen - oldLen;
			}
		}
		delete[] swap;
		return;
	}
	//generate a route
	void exec(const double** curInfo = nullptr) {
		int source = rand() % cityNumbers;
		forwardPath[0] = backwardPath[0] = source;
		int forwardPtr = 1, backwardPtr = 1;
		int ant1 = source, ant2 = source, newLoc;
		init();
		unvisited[source] = unvisited[cityPtr--];
		while (cityPtr >= 0) {
			if (cityPtr >= 0) {
				newLoc = selectCity(ant1, curInfo);
				forwardPath[forwardPtr++] = newLoc;
				ant1 = newLoc;
			}
			if (cityPtr >= 0) {
				newLoc = selectCity(ant2, curInfo);
				backwardPath[backwardPtr++] = newLoc;
				ant2 = newLoc;
			}
		}
		int pathPtr = 0;
		for (pathPtr = 0; pathPtr < forwardPtr; pathPtr++)
			path[pathPtr] = forwardPath[pathPtr];

		for (int i = backwardPtr - 1; i > 0; i--) {
			path[pathPtr++] = backwardPath[i];
		}
		path[pathPtr] = path[0];
		this->totalLen = getTotalLen(path);
		path[cityNumbers + 1] = this->totalLen;
		opt2(path, rand() % MAX_OPT2_TIMES);
	}
	//some util function for information exchange
	void  output(int* target) {
		for (int i = 0; i <= cityNumbers; i++)
			target[i] = path[i];
		target[cityNumbers + 1] = this->totalLen;
		target[cityNumbers + 2] = getTag(this->path);
	}
	inline static int getTag(int* path) {
		int tag = 0;
		for (int i = 0; i < cityNumbers; i++) {
			tag += path[i];
			tag = tag << 1;
			tag = tag % INF;
		}
		return tag;
	}
	inline static int getTotalLen(int* path) {
		int totalLen = 0;
		for (int i = 0; i < cityNumbers; i++) {
			totalLen += cityMap[path[i]][path[i + 1]];
		}
		return totalLen;
	}

	Ant() {
		path = new int[cityNumbers + 2];
		unvisited = new int[cityNumbers + 1];
		forwardPath = new int[cityNumbers + 1];
		backwardPath = new int[cityNumbers + 1];
		init();
	}
	~Ant() {
		delete[] path;
		delete[] unvisited;
		delete[] forwardPath;
		delete[] backwardPath;
	}
};

//simulate an colony
class Nest {
protected:
	//if dir=1,update when the len of data<target
	//if dir=-1,update when the len of data>target
	void replace(int antNumber, int*& target, int dir) {
		if (ants[antNumber].totalLen * dir < target[cityNumbers + 1] * dir) {
			ants[antNumber].output(target);
			return;
		}
	}
public:
	Ant* ants;
	int* bestSofar;
	int* curBest;
	int* curWorst;
	double wallExecTime;//!
	Nest() {
		ants = new Ant[maxAntNumber];
		bestSofar = new int[cityNumbers + 3];
		bestSofar[cityNumbers + 1] = INF;
		curBest = new int[cityNumbers + 3];
		curWorst = new int[cityNumbers + 3];
		wallExecTime = 1;//!
	}
	~Nest() {
		delete[] ants;
		delete[] bestSofar;
		delete[] curBest;
		delete[] curWorst;
	}
	void updateInfo(int* best, int* worst, int* globalBest, int times = 1) {
		for (int i = 0; i < times; i++) {
			phermoreVolatile(threadNum);
			phermorePunish(worst, threadNum);
			phermoreMark(best, best[cityNumbers + 1], threadNum);
			phermoreMark(globalBest, globalBest[cityNumbers + 1], threadNum);
		}
		updateInfoMatrix();
	}

	//do local iteration for curIter times, and update phermore matrix if the individual==true
	void exec(int curIter = ANT_ITER, bool individual = true) {
		double start;
		start = MPI_Wtime();
		int bestLen = INF, worstLen = 0, bestAnt = 0, worstAnt = 0;
		curBest[cityNumbers + 1] = INF;
		curWorst[cityNumbers + 1] = 1;
		for (int iter = 0; iter < curIter; iter++) {
			bestAnt = worstAnt = 0;
			bestLen = INF;
			worstLen = 0;
#pragma omp parallel for num_threads(threadNum) if(antNumber/threadNum>8) schedule(static) firstprivate(curInfo)//proc_bind(close)
			for (int loop = 0; loop < antNumber; loop++)
				ants[loop].exec(curInfo);
			for (int i = 0; i < antNumber; i++) {
				if (ants[i].totalLen < bestLen) {
					bestLen = ants[i].totalLen;
					bestAnt = i;
				}
				if (ants[i].totalLen > worstLen) {
					worstLen = ants[i].totalLen;
					worstAnt = i;
				}
			}

			if (bestAnt == worstAnt) {
				//local optimal
				continue;
			}

			replace(bestAnt, bestSofar, 1);
			replace(bestAnt, curBest, 1);
			replace(worstAnt, curWorst, -1);
			//modify volatile

			if (individual)
				updateInfo(ants[bestAnt].path, ants[worstAnt].path, bestSofar, 1);
		}
		this->wallExecTime = MPI_Wtime() - start;
	}
	void exportPaths(int* paths) {
		int pathLen = cityNumbers + 3;
		copyIntArr(curBest, paths, pathLen);
		copyIntArr(curWorst, paths + pathLen, pathLen);
	}
	void updateBest(int* outCurBest) {
		if (outCurBest[cityNumbers + 1] < bestSofar[cityNumbers + 1])
			copyIntArr(outCurBest, bestSofar, FULL_PATH_LEN);
	}
	void inputPaths(int* outCurBest, int* outCurWorst, int* enableList) {
		if (foreignWeight == 0)
			return;
		for (int i = 0; i < foreignWeight; i++) {
			phermoreVolatile(threadNum);
			if (enableList[1])
				phermorePunish(outCurWorst, threadNum);
			if (enableList[0])
				phermoreMark(outCurBest, threadNum);
		}
		updateBest(outCurBest);
	}
	void inputPaths(int* paths, int* enableList) {
		inputPaths(paths, paths + FULL_PATH_LEN, enableList);
	}

};


//----------------------------------mpi codes--------------------------------------

//it performs on a 2 path array with lengh 3*(cityNumber+3)
//only reserves best and worst route so far
void op_cmpBestAndWorstVerbose(int* in, int* inout, int* len, MPI_Datatype* dptr) {
	if (in[cityNumbers + 1] < inout[cityNumbers + 1])
		copyIntArr(in, inout, FULL_PATH_LEN);
	if (in[FULL_PATH_LEN + cityNumbers + 1] > inout[FULL_PATH_LEN + cityNumbers + 1])
		copyIntArr(in + FULL_PATH_LEN, inout + FULL_PATH_LEN, FULL_PATH_LEN);
}

void treeBcast(int* target, MPI_Datatype curType, int curMaster = 0) {
	int inteval = 0;
	bool hasUpdated = false;
	int newInteval = 0;
	MPI_Status status;
	int myShiftedID = myid - curMaster >= 0 ? myid - curMaster : myid - curMaster + numprocs;
	if (myid == curMaster) {
		inteval = numprocs / 2 + numprocs % 2;
		MPI_Send(target, 1, curType, (curMaster + inteval) % numprocs, numprocs - 1, MPI_COMM_WORLD);
		inteval = inteval - 1;
	}
	else
	{
		MPI_Recv(target, 1, curType, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		inteval = status.MPI_TAG;
	}

	while (true) {
		if (inteval == myShiftedID)
			break;
		if (inteval == myShiftedID + 1) {
			MPI_Send(target, 1, curType, (myid + 1) % numprocs, inteval, MPI_COMM_WORLD);
			break;
		}
		newInteval = (inteval + myShiftedID) / 2 + (inteval + myShiftedID) % 2;
		MPI_Send(target, 1, curType, (newInteval + curMaster) % numprocs, inteval, MPI_COMM_WORLD);
		inteval = newInteval - 1;
	}
}
//A common interface
void Bcast(int* target, MPI_Datatype curType, int curMaster = 0) {
	treeBcast(target, curType, curMaster);
}

class MpiComm
{
public:
	MpiComm(int argc, char* argv[]) {
		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &myid);
		MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	}
	~MpiComm() {
		MPI_Finalize();
	}
public:
	void init_sync_paras(int* paras, int num) {
		if (numprocs < 2)
			return;
		MPI_Bcast(paras, num, MPI_INT, 0, MPI_COMM_WORLD);
	}
	void init_sync_cityMap() {
		if (numprocs < 2)
			return;
		MPI_Datatype int2DArr;
		MPI_Type_vector(cityNumbers, cityNumbers, cityNumbers + 1, MPI_INT, &int2DArr);
		MPI_Type_commit(&int2DArr);
		Bcast(*cityMap, int2DArr, 0);
		MPI_Type_free(&int2DArr);
	}

	void commResVerbose(Nest& nest, bool doUpdate = true) {
		if (numprocs < 2)
			return;
		MPI_Datatype doublePath;
		MPI_Type_contiguous(2 * FULL_PATH_LEN, MPI_INT, &doublePath);
		MPI_Type_commit(&doublePath);
		MPI_Op myop;
		MPI_Op_create((MPI_User_function*)op_cmpBestAndWorstVerbose, 1, &myop);
		int* tPath = new int[FULL_PATH_LEN * 2];
		nest.exportPaths(tPath);
		int* tPathRes = new int[FULL_PATH_LEN * 2]{};

		MPI_Reduce(tPath, tPathRes, 1, doublePath, myop, 0, MPI_COMM_WORLD);
		if (doUpdate) {
			Bcast(tPathRes, doublePath);
			nest.inputPaths(tPathRes, updateList);
		}
		else if (myid == 0) {
			nest.updateBest(tPathRes);
		}

		delete[] tPath;
		delete[] tPathRes;
		MPI_Type_free(&doublePath);
		MPI_Op_free(&myop);
	}

};
//----------------------------------other codes--------------------------------------
int main(int argc, char* argv[]) {
	bool onLinux = false;
	myid = 0;//preInit this just in case for the situation of 1 node
#ifdef __linux
	uint64_t wallStart = GetTimeStamp();
	onLinux = true;
#endif // __linux
	MpiComm comm(argc, argv);
	double start = MPI_Wtime();
	srand(myid);
	threadNum = omp_get_max_threads() > 4 ? 4 : omp_get_max_threads();
	if (myid == 0)
		printf("threadNum=%d\n", threadNum);
	maxIter = MAX_ITER;
	foreignWeight = FOREIGN_WEIGHT;
	antIter = ANT_ITER;
	int paras[3];
	if (myid == 0) {
		matGen(!USEINPUT);
		paras[0] = cityNumbers;
		comm.init_sync_paras(paras, 1);
	}
	else {
		comm.init_sync_paras(paras, 1);
		cityNumbers = paras[0];
		matInit();
	}
	restartInteval = cityNumbers * 3;
	comm.init_sync_cityMap();
	herusticInit(threadNum);
	pherInit(threadNum);
	updateInfoMatrix(threadNum);
	curInfo = (const double**)info;
	antNumber = int(cityNumbers*ANT_NUM_RATIO/numprocs);//int(cityNumbers * ANT_NUM_RATIO);
	maxAntNumber = antNumber;
	MPI_Barrier(MPI_COMM_WORLD);
	if (myid == 0)
		printf("totalNumber=%d,avg number=%d\n", antNumber * numprocs, antNumber);
	MPI_Barrier(MPI_COMM_WORLD);
	if (myid == 0)
		printf("nests has fully inited\n");
	//srand(myid);
	srand(time(nullptr));
	Nest nest;
	int ptr = 0;
	int* iters = new int[maxIter] {0};
	int* scores = new int[maxIter] {INF};
	int badTimes = 0;
#ifdef __linux
	uint64_t tmpStart = GetTimeStamp();
#endif // __linux
	for (int i = 0; i < maxIter; i++) {//
		if (true) {//i % 5
			nest.exec(antIter);
			comm.commResVerbose(nest);
		}

		if (scores[ptr] > nest.bestSofar[cityNumbers + 1]) {
			ptr++;
			iters[ptr] = i;
			scores[ptr] = nest.bestSofar[cityNumbers + 1];
			badTimes = 0;
		}
		else {
			badTimes++;
			if (badTimes > restartInteval) {
				restart();
				srand(time(nullptr));
				badTimes = 0;
			}
		}

	}

	comm.commResVerbose(nest, false);

	if (myid == 0 && onLinux==false) {
		printf("time=%f\n", MPI_Wtime() - start);
	}

#ifdef __linux
	if (myid == 0)
		printf("Wall time: %ld us,commTime: %d\n", ((uint64_t)(GetTimeStamp() - wallStart)), ((uint64_t)(GetTimeStamp() - tmpStart)) / maxIter);
#endif // __linux

	if (myid == 0) {
		printf("calculation finished\nShortest path length so far is %d\n", nest.bestSofar[cityNumbers + 1]);
		printf("corresponding path is: ");
		printPath(nest.bestSofar, cityNumbers);
		printf("decline is\n");
		for (int i = 1; i <= ptr; i++)
			printf("%d %d \n", iters[i], scores[i]);
	}

	delete[] iters;
	delete[] scores;
	return 0;
}


