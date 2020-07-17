#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include<CL/cl.hpp>
#include<iostream>
#include<chrono>
#include<string>
#include<chrono>
#include<ctime>

using namespace std;

class GPUComponents {
public:
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;

	GPUComponents() {
		cout << "Przygotowuje GPU ..." << endl;

		// Pobieramy wektor platform
		vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		_ASSERT(platforms.size() > 0);
		cl::Platform platform;
		
		// Przeszukanie platform w poszukiwaniu urządzeń GPU
		for (int i = 0; i < platforms.size(); i++) {
			vector<cl::Device> devices;
			devices.clear();
			platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
			if (devices.size() > 0) {
				platform = platforms[i];
				break;
			}
		}

		// Nie znaleziono urządzeń GPU
		if (platform.getInfo<CL_PLATFORM_NAME>() == "") {
			cout << "Nie znaleziono urządzeń GPU";
			exit(-1);
		}

		vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		cl::Device device = devices.front();
		cout << "Nazwa platformy: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
		cout << "Wybrano urządzenie: " << device.getInfo<CL_DEVICE_VENDOR>() << device.getInfo<CL_DEVICE_NAME>() << endl;

		// Tworzymy kontekst zawierający jedno urządzenie
		context = cl::Context( device );



		// Opakowujemy kod kernel w obiekt Sources
		cl::Program::Sources sources;

		// kernel code

		string kernelCode =
			"__kernel void findPattern(__global char *number, __global char *pattern, __global int *patternSize, __global int *occurences){ "
			""
			"int myId = get_global_id(0);"
			"bool match = true;"
			"int j = 0;"
			"for(int i = myId; i < *patternSize + myId; i++) {"
			"if(number[i] != pattern[j]){"
			" match = false;"
			"}"
			"j++;"
			"}"
			"if (match == true) {"
			"atomic_inc(occurences);"
			"}"
			"} ";

		// -------------------------------------------------- //

		sources.push_back({ kernelCode.c_str(), kernelCode.length() });

		// Inicjalizujemy i budujemy kod programu
		program = cl::Program(context, sources);
		auto err = program.build({ device });
		if (err != 0) { exit(err); }

		// Tworzymy kolejkę do której bedziemy wysyłać komendy
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		cout << "GPU gotowe!" << endl << endl;
	}
};


// Klasa zawierająca przyjazne dla konkretnych urządzeń buffery zawierające wygenerowaną liczbę
class NumberBuffer {
public:
	char* cpu;
	cl::Buffer* gpu;

	NumberBuffer() {
		cpu = NULL;
		gpu = NULL;
	}
};


// Klasa zawierająca przyjazne dla konkretnych urządzeń buffery zawierające wzory do wyszukiwania
class PatternBuffer {
public:
	char* cpu;
	cl::Buffer* gpu;
	cl::Buffer* gpuPatternLength;

	PatternBuffer(string pattern, GPUComponents GPU, long long size) {

		// Sprawdzam czy długość wzoru jest mniejsza od ciągu liczb
		long long patternLength = pattern.length();
		if (patternLength > size) { cout << "Dlugosc wzoru wieksza od dlugosci liczby!" << endl; exit(-1); }


		// Kopiuję podane w konstruktorze wartości do odpowiednich bufforów
		cpu = new char[pattern.length() + 1];
		strcpy_s(cpu, patternLength + 1, pattern.c_str());

		gpu = new cl::Buffer(GPU.context, CL_MEM_READ_ONLY, sizeof(char) * patternLength);
		GPU.queue.enqueueWriteBuffer(*gpu, CL_TRUE, 0, sizeof(char) * patternLength, cpu);

		gpuPatternLength = new cl::Buffer(GPU.context, CL_MEM_READ_ONLY, sizeof(int));
		GPU.queue.enqueueWriteBuffer(*gpuPatternLength, CL_TRUE, 0, sizeof(int), &patternLength);
	}

};

NumberBuffer generateNumber(long long size, GPUComponents GPU);
void findPatternBMonCPU(string number, string pattern);
int max(int a, int b);
void findPatternOnGPU(GPUComponents GPU, cl::Buffer number, cl::Buffer pattern, cl::Buffer patternLength, long long size);

int main()
{

// Przygotowanie GPU
	GPUComponents GPU;
// -----------------

// Generowanie liczby
	long long size;
	cout << "Wybierz wielkosc liczby: ";
	cin >> size;

	NumberBuffer number = generateNumber(size, GPU);
// -------------------

// Przygotowanie wzorca w buforach na CPU i GPU
	string p;
	cout << "Podaj wzorzec do znalezienia w liczbie: ";
	cin >> p;

	PatternBuffer pattern(p, GPU, size);

// ---------------------------------------------

// Policzenie ilosci wystąpien wzroca w liczbie na CPU przy uzyciu algorytmu Boyera Moore'a
	findPatternBMonCPU(number.cpu, pattern.cpu);
// ----------------------------------------------------------------------------------------

// Policzenie ilosci wystapien wzorca w liczbie na GPU
	findPatternOnGPU(GPU, *number.gpu, *pattern.gpu, *pattern.gpuPatternLength, size);
// ---------------------------------------------------

	delete[] number.cpu;
	delete[] pattern.cpu;
	return 0;
}

NumberBuffer generateNumber(long long size, GPUComponents GPU) {
	cout << "Generuje liczbe ..." << endl;
	srand(time(NULL));
	NumberBuffer number;
	number.cpu = new char[size+1];
	for (long long i = 0; i < size; i++) {

		short d = (rand() * 123) % 10;

		if (size > 1000) {
			if (i % (size / 10) == 0) {
				cout << ((double(i) * 100.0) / double(size)) << "%" << endl;
			}
		}


		number.cpu[i] = char('0' + d);
	}

	number.cpu[size] = '\0';

	number.gpu = new cl::Buffer(GPU.context, CL_MEM_READ_ONLY, sizeof(char) * size);
	GPU.queue.enqueueWriteBuffer(*number.gpu, CL_TRUE, 0, sizeof(char) * size, number.cpu);
	cout << "Wygenerowalem liczbe o " << size << " znakach!" << endl;
	return number;
}

void findPatternOnGPU(GPUComponents GPU, cl::Buffer number, cl::Buffer pattern, cl::Buffer patternLength, long long size) {


	// buffor zmiennej w której zliczać będziemy wystąpienia
	cl::Buffer occurencesGPU(GPU.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(int));

	// Przypisujemy argumenty do naszej funkcji w kernelu
	cl::Kernel finding(GPU.program, "findPattern");
	finding.setArg(0, number);
	finding.setArg(1, pattern);
	finding.setArg(2, patternLength);
	finding.setArg(3, occurencesGPU);

	cl::Event event;

	cout << "Szukam na GPU ..." << endl;

	// Kolejkujemy nasz program wykorzystując tyle Work Item'ów jak długa jest liczba
	GPU.queue.enqueueNDRangeKernel(finding, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &event);

	int occurencesGPUint = 0;

	// Pobieramy ilość wystąpień z GPU
	GPU.queue.enqueueReadBuffer(occurencesGPU, CL_TRUE, 0, sizeof(int), &occurencesGPUint);

	cout << "Wystapenia na GPU : " << occurencesGPUint << endl;
	cl_ulong time_start, time_stop;
	event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
	event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_stop);


	cout << "Czas wykonania na GPU: " << (double)(time_stop - time_start) / 1000000000 << " sekund"; // dzielę by przekształcić nanosekundy na sekundy
}

void findPatternBMonCPU(string number, string pattern) {

	cout << "Szukam na CPU ..." << endl;

	long long numberLength = number.length();
	long long patternLength = pattern.length();
	

	// inicjalizujemy tablicę Last zawierającą wszystkie wykorzystywane znaki - liczby (indeksy od 0 do 9)
	long long* Last = new long long[10];
	for (int i = 0; i < 10; i++) Last[i] = -1;

	long long i = 0;
	long long j;
	long long occurencesCPU = 0;
	auto start = chrono::high_resolution_clock::now();

	// tablicę Last wypełniamy indeksami ostatnich wystąpień danych liczb ze wzoru
	for (int i = 0; i < patternLength; i++) Last[(int)pattern[i] - (int)'0'] = i;

	// zapętlamy dopóki okno wzoru mieści się w pozostałej do przeanalizowania części ciągu
	while (i <= numberLength - patternLength) {

		// poprawność wzoru z okienkiem sprawdzamy od tyłu
		j = patternLength - 1;

		// dopóki wzór zgadza się z okienkiem j zmniejszamy o 1
		while ((j > -1) && (pattern[j] == number[i + j])) j--;


		if (j == -1) {
			// jeśli j == -1 to oznacza to, że znaleźliśmy wystąpienie wzoru i przesuwamy wzór o 1
			occurencesCPU++;
			i++;
		}
		else {
			// w przeciwnym wypadku wzór przesuwamy o max( 1 , lub indeks ostatniego wystąpienia błędnej liczby we wzorze)
			i += max(1, j - Last[(int)number[i + j] - (int)'0']);
		}
	}

	auto stop = chrono::high_resolution_clock::now();


	cout <<"Wystapienia na CPU: "<< occurencesCPU <<endl;
	cout << "Czas wykonania na CPU: " << chrono::duration_cast<chrono::duration<double>>(stop - start).count() << " sekund"<<endl;

	cout << endl << endl;
	delete[] Last;
	return;
}


int max(int a, int b) {
	if (a >= b) return a;
	else return b;
}