/////////////////////
//Theoretically any physical node is OK
//the cpu number can be replaced with 16, 8 or 4 if 32 CPU consumes too much time in pending
//but cpu number is recommoned to be the pow of 2
//However, this program's binding method is designed for 32 CPU
/*
#!/bin/bash
#SBATCH --partition=physical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:01:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
module load gcc/10.3.0
g++ -fopenmp -lnuma -Wall -O3 -o yuzhouhuo_kseqalign yuzhouhuo_kseqalign.cpp
export OMP_PROC_BIND=true
export OMP_NESTED=TRUE
export OMP_MAX_ACTIVE_LEVELS=2
cat mseq-big13-example.dat | ./yuzhouhuo_kseqalign
rm yuzhouhuo_kseqalign
*/
/////////////////////
#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
#include <iostream>
#include <cstring>
#include<math.h>
#include "sha512.hh"
#include <thread>
#include <algorithm>
#ifdef __linux
#include<stdlib.h>
#include <sys/time.h>
#include <numa.h>
#endif
unsigned long long MAX_MEMORY_SIZE = 34359738368L;
constexpr auto MAX_THREAD_SIZE = 8 * 1024 * 1024;
using namespace std;
int pxy, pgap;
#ifdef __linux
uint64_t GetTimeStamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
}
#endif

//calculate the hash result of pair align
string pairHash(string& align1, string& align2)
{
	std::string align1hash = sw::sha512::calculate(align1);
	std::string align2hash = sw::sha512::calculate(align2);
	std::string problemhash = sw::sha512::calculate(align1hash.append(align2hash));
	return problemhash;
}

struct pairAlignProb
{
	int xloc = -1;
	int yloc = -1;
	int originLoc = -1;
	unsigned long long dpSize = 0;
};
bool pairProbCmp(const pairAlignProb& a, const pairAlignProb& b) {
	return a.dpSize > b.dpSize;
}

struct pairAlignRes
{
	std::string problemhash = "";
	int score = -1;
	pairAlignRes(std::string str, int s) { problemhash = str; score = s; }
	pairAlignRes(){}
};

int** pairMatInit(int m, int n, int threadNum = 1,bool fill=true) {
	int** dp = new int* [m + 1];
	size_t size = m + 1;
	size *= n + 1;
	int* dp0 = new int[size];
	if (!dp || !dp0)
	{
		std::cerr << "getMinimumPenalty: new failed" << std::endl;
		exit(1);
	}
	dp[0] = dp0;

	for (int i = 1; i < m + 1; i++) {
		dp[i] = dp[i - 1] + n + 1;
	}
	// intialising the table
#pragma omp parallel for num_threads(threadNum) schedule(static) proc_bind(close)
	for (int i = 1; i <= m; i++) {
		if(fill)
			memset(dp[i], -1, (n + 1) * 4);
		dp[i][0] = i * pgap;
	}
	dp[0][0] = 0;

	for (int j = 0; j <= n; j++)
	{
		dp[0][j] = j * pgap;
	}
	return dp;
}
//compare serially(used when only one thread available to task)
void getMinPairPenaltySerial(std::string& x, std::string& y, int** dp, int nthreads = 4)
{
	int i, j; // intialising variables
	int m = x.length(); // length of gene1
	int n = y.length(); // length of gene2
	int tmpMinPgap, tmpMinPxy = 0;
	// calcuting the minimum penalty
	for (i = 1; i <= m; i++)
	{
		for (j = 1; j <= n; j++)
		{
			if (x[i - 1] == y[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1];
			}
			else
			{
				tmpMinPgap = (dp[i - 1][j] < dp[i][j - 1] ? dp[i - 1][j] : dp[i][j - 1]) + pgap;
				tmpMinPxy = dp[i - 1][j - 1] + pxy;
				dp[i][j] = tmpMinPxy < tmpMinPgap ? tmpMinPxy : tmpMinPgap;
			}
		}
	}
	return;
}

void getMinPairPenaltyRowSmp(std::string& x, std::string& y, int** dp, int nthreads = 4) {
	int m = x.length(); // length of gene1
	int n = y.length(); // length of gene2
	int tmpMinPgap, tmpMinPxy = 0;
#pragma omp parallel num_threads(nthreads) shared(dp) private(tmpMinPgap,tmpMinPxy) proc_bind(close)
	{
		int currThreadNum = omp_get_thread_num();
		for (int i = 1 + currThreadNum; i <= m; i += nthreads)
		{
			for (int j = 1; j <= n; j++)
			{
				while (dp[i - 1][j] < 0){ std::this_thread::yield(); }
				if (x[i - 1] == y[j - 1])
				{
					dp[i][j] = dp[i - 1][j - 1];
				}
				else
				{
					tmpMinPgap = (dp[i - 1][j] < dp[i][j - 1] ? dp[i - 1][j] : dp[i][j - 1]) + pgap;
					tmpMinPxy = dp[i - 1][j - 1] + pxy;
					dp[i][j] = tmpMinPxy < tmpMinPgap ? tmpMinPxy : tmpMinPgap;
				}
			}
		}
	}
	return;
}

//generate pair align res from DP array
pairAlignRes getAlignedSeq(std::string& x, std::string& y, int** dp) {
	int m = x.length(); // length of gene1
	int n = y.length(); // length of gene2
	int l = n + m; // maximum possible length
	int i = m, j = n;
	int xpos = l;
	int ypos = l;
	int* xans = new int[m + n + 1];
	int* yans = new int[m + n + 1];
	//trace back
	while (!(i == 0 || j == 0))
	{
		if ((x[i - 1] == y[j - 1]) || dp[i - 1][j - 1] + pxy == dp[i][j])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = (int)y[j - 1];
			i--; j--;
		}
		else if (dp[i - 1][j] + pgap == dp[i][j])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = (int)'_';
			i--;
		}
		else if (dp[i][j - 1] + pgap == dp[i][j])
		{
			xans[xpos--] = (int)'_';
			yans[ypos--] = (int)y[j - 1];
			j--;
		}
	}
	//deal with the gap in the sequence start
	while (i > 0) {
		xans[xpos--] = (int)x[--i];
		yans[ypos--] = (int)'_';
	}
	while (j > 0) {
		xans[xpos--] = (int)'_';
		yans[ypos--] = (int)y[--j];
	}
	xpos++;
	ypos++;
	int id = xpos > ypos ? ypos : xpos;
	std::string align1 = "";
	std::string align2 = "";
	for (int a = id; a <= l; a++)
	{
		char tmp = (char)xans[a];
		if (tmp > 32)
			align1.append(1, tmp);
	}
	for (int a = id; a <= l; a++)
	{
		char tmp = (char)yans[a];
		if (tmp > 32)
			align2.append(1, tmp);
	}
	delete[] xans;
	delete[] yans;
	pairAlignRes res(pairHash(align1, align2), dp[m][n]);
	return res;
}

//align one pair(include init, DP, and align result generation)
pairAlignRes pairAlign(std::string& x, std::string& y, int threadNum = 1)
{
	int** dp;
	if (threadNum > 1){
		dp = pairMatInit(x.length(), y.length(),threadNum,true);
		getMinPairPenaltyRowSmp(x, y, dp, threadNum);
	}else{
		dp = pairMatInit(x.length(), y.length(),1,false);
		getMinPairPenaltySerial(x, y, dp);
	}pairAlignRes res = getAlignedSeq(x, y, dp);
	delete[] dp[0];
	delete[] dp;
	return res;
}

//just an interface
void inline pairAlignChunk(string* genes, string* alignHashs, pairAlignProb* problems, int* penalties, int loc, int threadNum)
{
	pairAlignRes res = pairAlign(genes[problems[loc].xloc], genes[problems[loc].yloc], threadNum);
	alignHashs[problems[loc].originLoc] = res.problemhash;
	penalties[problems[loc].originLoc] = res.score;
}

void batchExec(unsigned long long availableSize, int numPairs, std::string* genes, \
	pairAlignProb* problems, string* alignHashs, int* penalties) {
	int maxThreads = omp_get_max_threads();
	int maxBatch = int(availableSize / problems[numPairs - 1].dpSize);
	int maxPossibleIters = maxBatch > maxThreads ? maxThreads : maxBatch;
	int* iterArr = new int[maxPossibleIters+1] { numPairs };
	int* batchSizeArr = new int[maxPossibleIters+1] { 1 };
	int iterPtr = 0, probPtr = 0, batchSize = 1;
	unsigned long long tmpMaxProbSize = availableSize / 1;
	//split task according to their size and try to parallel as much task as possible
	while ((maxThreads / batchSize > 0) && probPtr < numPairs)
	{
		while ((probPtr < numPairs) && (problems[probPtr].dpSize > tmpMaxProbSize)) { probPtr += batchSize; }
		probPtr = probPtr > numPairs ? numPairs : probPtr;
		iterArr[iterPtr] = probPtr;
		batchSizeArr[iterPtr] = batchSize;
		iterPtr++;
		batchSize *= 2;
		tmpMaxProbSize = availableSize / batchSize;
	}
	iterArr[iterPtr]=numPairs;
	//use all threads for tasks that too big to run with others
	for (int i = 0; i < iterArr[1]; i++)
		pairAlignChunk(genes, alignHashs, problems, penalties, i, maxThreads>16?16:maxThreads);
	//nested parallel to spread threads among individual tasks
	for (int i = 1; i < iterPtr; i++) {
#pragma omp parallel for schedule(guided) num_threads(batchSizeArr[i]) proc_bind(spread)
		for (int j = iterArr[i]; j < iterArr[i+1]; j++)
			pairAlignChunk(genes, alignHashs, problems, penalties, j, maxThreads / batchSizeArr[i]);
	}
}

void getMinimumPenalties(std::string* genes, int k) {
	uint64_t start = GetTimeStamp();
/*
	if(k<=2){
		pairAlignRes tmpRes=pairAlign(genes[1], genes[0], omp_get_max_threads());
		printf("Time: %ld us\n", ((uint64_t)(GetTimeStamp() - start)));
		std::cout<<sw::sha512::calculate(string("").append(tmpRes.problemhash))<<std::endl;
		std::cout<<tmpRes.score<<' '<<std::endl;
		return;
	}
*/
	int numPairs = k * (k - 1) / 2;
	if (numPairs < 3)
		return;//"wrong input"
	int* penalties = new int[numPairs];
	string* alignHashs = new string[numPairs];
	string resHash = "";
	pairAlignProb* problems = new pairAlignProb[numPairs];
	int probPtr = 0;
	//create task list and sort them according to size
	for (int i = 1; i < k; i++) {
		for (int j = 0; j < i; j++) {
			problems[probPtr].xloc = i;
			problems[probPtr].yloc = j;
			problems[probPtr].originLoc = probPtr;
			problems[probPtr].dpSize = 4 * (static_cast<unsigned long long>(genes[i].length()) + 1) \
				* (static_cast<unsigned long long>(genes[j].length() + 1));
			probPtr++;
		}
	}
	sort(problems, problems + numPairs, pairProbCmp);
	//calculate the maximum size available, the 0.9 is for safety
	unsigned long long availableSize =
		(MAX_MEMORY_SIZE - (numPairs * static_cast<unsigned long long>\
			(2 * sizeof(int) + sizeof(pairAlignProb)) + sizeof(genes))) * 0.9;
	batchExec(availableSize, numPairs, genes, problems, alignHashs, penalties);
	for (int i = 0; i < numPairs; i++) {
		resHash = sw::sha512::calculate(resHash.append(alignHashs[i]));
	}

	printf("Time: %ld us\n", ((uint64_t)(GetTimeStamp() - start)));
	cout<<resHash<<endl;
	for (int i=0;i<numPairs;i++){
		std::cout<<penalties[i]<<' ';
	}
	std::cout<<std::endl;
	delete[] problems;
	delete[] penalties;
	delete[] alignHashs;
	//return resHash;
	return;
}

int main() {
	omp_set_nested(2);
	omp_set_max_active_levels(2);
	int k;
	std::cin >> pxy;
	std::cin >> pgap;
	std::cin >> k;
	std::string* genes = new string[k];
	for (int i = 0; i < k; i++) std::cin >> genes[i];
	string res = "";
	getMinimumPenalties(genes, k);
	delete[] genes;
}

