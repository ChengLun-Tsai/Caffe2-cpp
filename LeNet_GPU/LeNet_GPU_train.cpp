#include <iostream>

#include "caffe2/core/flags.h"		// CAFFE2_DECLARE_xxx
#include "caffe2/core/init.h"		// GlobalInit
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/utility_ops.h"

#include "time.h"

#define PROTO_DB_PATH "./mnist-train-nchw-leveldb"
#define DEVICE_TYPE CUDA 
#define TENSORTYPE TensorCUDA
#define BATCH_SIZE 128

using namespace caffe2;
NetDef getLeNetDef();
NetDef getLeNetDef_Init();
void addTrainingOperators(NetDef& net_def);
/************************************************************/
/* Target: All training process happens on GPU side without */
/*		CPU Intervening.									*/
/* LeNet GPU version:										*/
/*   1. Use TensorCUDA to feed data into the network.		*/
/*   2. Make all operator in the network into CUDA version.	*/
/************************************************************/

int main(int argc, char** argv) {	
	// Initialize caffe2
	caffe2::GlobalInit(&argc, &argv);
	
	// Check GPU exists
	bool hasGPU = HasCudaGPU();
	if (hasGPU)
		std::cout << "Found GPU." << std::endl;
	else {
		std::cout << "GPU not found." << std::endl;
		return 0;
	}
	
	Workspace ws;
	NetDef leNetDef = getLeNetDef();
	addTrainingOperators(leNetDef);
	NetDef leNetDef_Init = getLeNetDef_Init();
	ws.RunNetOnce(leNetDef_Init);
	NetBase* leNetBase =  ws.CreateNet(leNetDef, false);
	clock_t start = clock();
	for (int i=1; i <= 1000; i++) {
		ws.RunNet("LeNet");
		if (i % 100 == 0) {
			Blob* accBlob = ws.GetBlob("accuracy");
			Blob* lossBlob = ws.GetBlob("loss");
			TensorCPU accTensor(accBlob->Get<TENSORTYPE>());
			TensorCPU lossTensor(lossBlob->Get<TENSORTYPE>());
			std::cout << "Training Iteration: " << i << " ";
			printf("Accuracy: %.2f ", accTensor.data<float>()[0]);
			printf("Loss: %.5f ", lossTensor.data<float>()[0]);
			printf("Time: %.1fs\n", (double)(clock() - start) / CLOCKS_PER_SEC);
		}
	}
	
	// This is to allow us to use memory leak checks.
	google::protobuf::ShutdownProtobufLibrary();
	
	return 0;
}

NetDef getLeNetDef() {
	NetDef net_def;
	net_def.set_name("LeNet");
	// db input
	OperatorDef tensorDbInput_Opdef;
	tensorDbInput_Opdef.set_type("TensorProtosDBInput");
	tensorDbInput_Opdef.add_input("dbreader_leveldb");
	tensorDbInput_Opdef.add_output("data_uint8_cuda");
	tensorDbInput_Opdef.add_output("label_cuda");
	tensorDbInput_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("batch_size", BATCH_SIZE, &tensorDbInput_Opdef);
	net_def.add_op()->CopyFrom(tensorDbInput_Opdef);
	net_def.add_external_input("dbreader_leveldb");
	
	// db input
	// OperatorDef tensorDbInput_Opdef;
	// tensorDbInput_Opdef.set_type("TensorProtosDBInput");
	// tensorDbInput_Opdef.add_input("dbreader_leveldb");
	// tensorDbInput_Opdef.add_output("data_uint8");
	// tensorDbInput_Opdef.add_output("label");
	// AddArgument<int>("batch_size", BATCH_SIZE, &tensorDbInput_Opdef);
	// net_def.add_op()->CopyFrom(tensorDbInput_Opdef);
	// net_def.add_external_input("dbreader_leveldb");
	
	// OperatorDef CPU_To_GPU_Data_Opdef;
	// CPU_To_GPU_Data_Opdef.set_type("CopyCPUToGPU");
	// CPU_To_GPU_Data_Opdef.add_input("data_uint8");
	// CPU_To_GPU_Data_Opdef.add_output("data_uint8_cuda");
	// CPU_To_GPU_Data_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	// net_def.add_op()->CopyFrom(CPU_To_GPU_Data_Opdef);
	
	// OperatorDef CPU_To_GPU_Label_Opdef;
	// CPU_To_GPU_Label_Opdef.set_type("CopyCPUToGPU");
	// CPU_To_GPU_Label_Opdef.add_input("label");
	// CPU_To_GPU_Label_Opdef.add_output("label_cuda");
	// CPU_To_GPU_Label_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	// net_def.add_op()->CopyFrom(CPU_To_GPU_Label_Opdef);
	
	OperatorDef cast_Opdef;
	cast_Opdef.set_type("Cast");
	cast_Opdef.add_input("data_uint8_cuda");
	cast_Opdef.add_output("data");
	cast_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("to", 1, &cast_Opdef);
	net_def.add_op()->CopyFrom(cast_Opdef);
	
	OperatorDef scale_Opdef;
	scale_Opdef.set_type("Scale");
	scale_Opdef.add_input("data");
	scale_Opdef.add_output("data");
	scale_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<float>("scale", float(1. / 256), &scale_Opdef);
	net_def.add_op()->CopyFrom(scale_Opdef);
	
	OperatorDef stopGradient_Opdef;
	stopGradient_Opdef.set_type("StopGradient");
	stopGradient_Opdef.add_input("data");
	stopGradient_Opdef.add_output("data");
	stopGradient_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(stopGradient_Opdef);
	
	// conv1
	OperatorDef conv1_Opdef;
	conv1_Opdef.set_type("Conv");
	conv1_Opdef.add_input("data");
	conv1_Opdef.add_input("conv1_w");
	conv1_Opdef.add_input("conv1_b");
	conv1_Opdef.add_output("conv1");
	conv1_Opdef.set_engine("CUDNN");
	conv1_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 5, &conv1_Opdef);
	net_def.add_op()->CopyFrom(conv1_Opdef);
	net_def.add_external_input("conv1_w");
	net_def.add_external_input("conv1_b");
	
	// pool1
	OperatorDef pool1_Opdef;
	pool1_Opdef.set_type("MaxPool");
	pool1_Opdef.add_input("conv1");
	pool1_Opdef.add_output("pool1");
	pool1_Opdef.set_engine("CUDNN");
	pool1_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 2, &pool1_Opdef);
	AddArgument<int>("stride", 2, &pool1_Opdef);
	net_def.add_op()->CopyFrom(pool1_Opdef);
	
	// conv2
	OperatorDef conv2_Opdef;
	conv2_Opdef.set_type("Conv");
	conv2_Opdef.add_input("pool1");
	conv2_Opdef.add_input("conv2_w");
	conv2_Opdef.add_input("conv2_b");
	conv2_Opdef.add_output("conv2");
	conv2_Opdef.set_engine("CUDNN");
	conv2_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 5, &conv2_Opdef);
	net_def.add_op()->CopyFrom(conv2_Opdef);
	net_def.add_external_input("conv2_w");
	net_def.add_external_input("conv2_b");
	
	// pool2
	OperatorDef pool2_Opdef;
	pool2_Opdef.set_type("MaxPool");
	pool2_Opdef.add_input("conv2");
	pool2_Opdef.add_output("pool2");
	pool2_Opdef.set_engine("CUDNN");
	pool2_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 2, &pool2_Opdef);
	AddArgument<int>("stride", 2, &pool2_Opdef);
	net_def.add_op()->CopyFrom(pool2_Opdef);
	
	// fc3
	OperatorDef fc3_Opdef;
	fc3_Opdef.set_type("FC");
	fc3_Opdef.add_input("pool2");
	fc3_Opdef.add_input("fc3_w");
	fc3_Opdef.add_input("fc3_b");
	fc3_Opdef.add_output("fc3");
	fc3_Opdef.set_engine("CUDNN");
	fc3_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(fc3_Opdef);
	net_def.add_external_input("fc3_w");
	net_def.add_external_input("fc3_b");
	
	// relu3
	OperatorDef relu3_Opdef;
	relu3_Opdef.set_type("Relu");
	relu3_Opdef.add_input("fc3");
	relu3_Opdef.add_output("relu3");
	relu3_Opdef.set_engine("CUDNN");
	relu3_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(relu3_Opdef);
	
	// pred(fc4)
	OperatorDef pred_Opdef;
	pred_Opdef.set_type("FC");
	pred_Opdef.add_input("relu3");
	pred_Opdef.add_input("pred_w");
	pred_Opdef.add_input("pred_b");
	pred_Opdef.add_output("pred");
	pred_Opdef.set_engine("CUDNN");
	pred_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(pred_Opdef);
	net_def.add_external_input("pred_w");
	net_def.add_external_input("pred_b");
	
	// softmax
	OperatorDef softmax_Opdef;
	softmax_Opdef.set_type("Softmax");
	softmax_Opdef.add_input("pred");
	softmax_Opdef.add_output("softmax");
	softmax_Opdef.set_engine("CUDNN");
	softmax_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(softmax_Opdef);

	return net_def;
}

NetDef getLeNetDef_Init() {
	NetDef net_def;
	net_def.set_name("LeNet_Init");
	// ONE
	OperatorDef one_Opdef;
	one_Opdef.set_type("ConstantFill");
	one_Opdef.add_output("ONE");
	AddArgument<std::vector<int>>("shape", vector<int>{1}, &one_Opdef);
	AddArgument<float>("value", 1.0, &one_Opdef);
	one_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(one_Opdef);

	// iter
	OperatorDef iter_Opdef;
	iter_Opdef.set_type("ConstantFill");
	iter_Opdef.add_output("iter");
	AddArgument<std::vector<int>>("shape", vector<int>{1}, &iter_Opdef);
	AddArgument<int>("value", 0, &iter_Opdef);
	AddArgument<int>("dtype", 10, &iter_Opdef);
	net_def.add_op()->CopyFrom(iter_Opdef);

	// dbreader_leveldb
	OperatorDef db_Opdef;
	db_Opdef.set_type("CreateDB");
	db_Opdef.add_output("dbreader_leveldb");
	AddArgument<std::string>("db_type", "leveldb", &db_Opdef);
	AddArgument<std::string>("db", PROTO_DB_PATH, &db_Opdef);
	net_def.add_op()->CopyFrom(db_Opdef);
	
	// conv1_w
	OperatorDef conv1_w_Opdef;
	conv1_w_Opdef.set_type("XavierFill");
	conv1_w_Opdef.add_output("conv1_w");
	AddArgument<std::vector<int>>("shape", vector<int>{20, 1, 5, 5}, &conv1_w_Opdef);
	conv1_w_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv1_w_Opdef);
	
	// conv1_b
	OperatorDef conv1_b_Opdef;
	conv1_b_Opdef.set_type("ConstantFill");
	conv1_b_Opdef.add_output("conv1_b");
	AddArgument<std::vector<int>>("shape", vector<int>{20}, &conv1_b_Opdef);
	conv1_b_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv1_b_Opdef);
	
	// conv2_w
	OperatorDef conv2_w_Opdef;
	conv2_w_Opdef.set_type("XavierFill");
	conv2_w_Opdef.add_output("conv2_w");
	AddArgument<std::vector<int>>("shape", vector<int>{50, 20, 5, 5}, &conv2_w_Opdef);
	conv2_w_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv2_w_Opdef);
	
	// conv2_b
	OperatorDef conv2_b_Opdef;
	conv2_b_Opdef.set_type("ConstantFill");
	conv2_b_Opdef.add_output("conv2_b");
	AddArgument<std::vector<int>>("shape", vector<int>{50}, &conv2_b_Opdef);
	conv2_b_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv2_b_Opdef);
	
	// fc3_w
	OperatorDef fc3_w_Opdef;
	fc3_w_Opdef.set_type("XavierFill");
	fc3_w_Opdef.add_output("fc3_w");
	AddArgument<std::vector<int>>("shape", vector<int>{500, 800}, &fc3_w_Opdef);
	fc3_w_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(fc3_w_Opdef);
	
	// fc3_b
	OperatorDef fc3_b_Opdef;
	fc3_b_Opdef.set_type("XavierFill");
	fc3_b_Opdef.add_output("fc3_b");
	AddArgument<std::vector<int>>("shape", vector<int>{500}, &fc3_b_Opdef);
	fc3_b_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(fc3_b_Opdef);
	
	// pred_w
	OperatorDef pred_w_Opdef;
	pred_w_Opdef.set_type("XavierFill");
	pred_w_Opdef.add_output("pred_w");
	AddArgument<std::vector<int>>("shape", vector<int>{10, 500}, &pred_w_Opdef);
	pred_w_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(pred_w_Opdef);
	
	// pred_b
	OperatorDef pred_b_Opdef;
	pred_b_Opdef.set_type("ConstantFill");
	pred_b_Opdef.add_output("pred_b");
	AddArgument<std::vector<int>>("shape", vector<int>{10}, &pred_b_Opdef);
	pred_b_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(pred_b_Opdef);
	

	return net_def;
}

void addTrainingOperators(NetDef& net_def) {
	// xent (CrossEntropy)
	OperatorDef xent_Opdef;
	xent_Opdef.set_type("LabelCrossEntropy");
	xent_Opdef.add_input("softmax");
	xent_Opdef.add_input("label_cuda");
	xent_Opdef.add_output("xent");
	xent_Opdef.set_engine("CUDNN");
	xent_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(xent_Opdef);
	
	// loss
	OperatorDef loss_Opdef;
	loss_Opdef.set_type("AveragedLoss");
	loss_Opdef.add_input("xent");
	loss_Opdef.add_output("loss");
	loss_Opdef.set_engine("CUDNN");
	loss_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(loss_Opdef);
	
	// accuracy
	OperatorDef accuracy_Opdef;
	accuracy_Opdef.set_type("Accuracy");
	accuracy_Opdef.add_input("softmax");
	accuracy_Opdef.add_input("label_cuda");
	accuracy_Opdef.add_output("accuracy");
	accuracy_Opdef.set_engine("CUDNN");
	accuracy_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(accuracy_Opdef);

	// loss_autogen_grad
	OperatorDef loss_autogen_grad_Opdef;
	loss_autogen_grad_Opdef.set_type("ConstantFill");
	loss_autogen_grad_Opdef.add_input("loss");
	loss_autogen_grad_Opdef.add_output("loss_autogen_grad");
	loss_autogen_grad_Opdef.set_engine("CUDNN");
	loss_autogen_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<float>("value", 1.0, &loss_autogen_grad_Opdef);
	net_def.add_op()->CopyFrom(loss_autogen_grad_Opdef);
	
	/************************************************************/
	/*					Add Gradient Operators 					*/
	/************************************************************/
	// loss grad
	OperatorDef loss_grad_Opdef;
	loss_grad_Opdef.set_type("AveragedLossGradient");
	loss_grad_Opdef.add_input("xent");
	loss_grad_Opdef.add_input("loss_autogen_grad");
	loss_grad_Opdef.add_output("xent_grad");
	loss_grad_Opdef.set_engine("CUDNN");
	loss_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(loss_grad_Opdef);
	
	// xent grad
	OperatorDef xent_grad_Opdef;
	xent_grad_Opdef.set_type("LabelCrossEntropyGradient");
	xent_grad_Opdef.add_input("softmax");
	xent_grad_Opdef.add_input("label_cuda");
	xent_grad_Opdef.add_input("xent_grad");
	xent_grad_Opdef.add_output("softmax_grad");
	xent_grad_Opdef.set_engine("CUDNN");
	xent_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(xent_grad_Opdef);
	
	// softmax grad
	OperatorDef softmax_grad_Opdef;
	softmax_grad_Opdef.set_type("SoftmaxGradient");
	softmax_grad_Opdef.add_input("softmax");
	softmax_grad_Opdef.add_input("softmax_grad");
	softmax_grad_Opdef.add_output("pred_grad");
	softmax_grad_Opdef.set_engine("CUDNN");
	softmax_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(softmax_grad_Opdef);

	// pred grad
	OperatorDef pred_grad_Opdef;
	pred_grad_Opdef.set_type("FCGradient");
	pred_grad_Opdef.add_input("relu3");
	pred_grad_Opdef.add_input("pred_w");
	pred_grad_Opdef.add_input("pred_grad");
	pred_grad_Opdef.add_output("pred_w_grad");
	pred_grad_Opdef.add_output("pred_b_grad");
	pred_grad_Opdef.add_output("relu3_grad");
	pred_grad_Opdef.set_engine("CUDNN");
	pred_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(pred_grad_Opdef);
	
	// relu3 grad
	OperatorDef relu3_grad_Opdef;
	relu3_grad_Opdef.set_type("ReluGradient");
	relu3_grad_Opdef.add_input("relu3");
	relu3_grad_Opdef.add_input("relu3_grad");
	relu3_grad_Opdef.add_output("fc3_grad");
	relu3_grad_Opdef.set_engine("CUDNN");
	relu3_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(relu3_grad_Opdef);
	
	// fc3 grad
	OperatorDef fc3_grad_Opdef;
	fc3_grad_Opdef.set_type("FCGradient");
	fc3_grad_Opdef.add_input("pool2");
	fc3_grad_Opdef.add_input("fc3_w");
	fc3_grad_Opdef.add_input("fc3_grad");
	fc3_grad_Opdef.add_output("fc3_w_grad");
	fc3_grad_Opdef.add_output("fc3_b_grad");
	fc3_grad_Opdef.add_output("pool2_grad");
	fc3_grad_Opdef.set_engine("CUDNN");
	fc3_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(fc3_grad_Opdef);
	
	// pool2 grad
	OperatorDef pool2_grad_Opdef;
	pool2_grad_Opdef.set_type("MaxPoolGradient");
	pool2_grad_Opdef.add_input("conv2");
	pool2_grad_Opdef.add_input("pool2");
	pool2_grad_Opdef.add_input("pool2_grad");
	pool2_grad_Opdef.add_output("conv2_grad");
	pool2_grad_Opdef.set_engine("CUDNN");
	pool2_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 2, &pool2_grad_Opdef);
	AddArgument<int>("stride", 2, &pool2_grad_Opdef);
	net_def.add_op()->CopyFrom(pool2_grad_Opdef);
	
	// conv2 grad
	OperatorDef conv2_grad_Opdef;
	conv2_grad_Opdef.set_type("ConvGradient");
	conv2_grad_Opdef.add_input("pool1");
	conv2_grad_Opdef.add_input("conv2_w");
	conv2_grad_Opdef.add_input("conv2_grad");
	conv2_grad_Opdef.add_output("conv2_w_grad");
	conv2_grad_Opdef.add_output("conv2_b_grad");
	conv2_grad_Opdef.add_output("pool1_grad");
	conv2_grad_Opdef.set_engine("CUDNN");
	conv2_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 5, &conv2_grad_Opdef);
	net_def.add_op()->CopyFrom(conv2_grad_Opdef);
	
	// pool1 grad
	OperatorDef pool1_grad_Opdef;
	pool1_grad_Opdef.set_type("MaxPoolGradient");
	pool1_grad_Opdef.add_input("conv1");
	pool1_grad_Opdef.add_input("pool1");
	pool1_grad_Opdef.add_input("pool1_grad");
	pool1_grad_Opdef.add_output("conv1_grad");
	pool1_grad_Opdef.set_engine("CUDNN");
	pool1_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 2, &pool1_grad_Opdef);
	AddArgument<int>("stride", 2, &pool1_grad_Opdef);
	net_def.add_op()->CopyFrom(pool1_grad_Opdef);
	
	// conv1 grad
	OperatorDef conv1_grad_Opdef;
	conv1_grad_Opdef.set_type("ConvGradient");
	conv1_grad_Opdef.add_input("data");
	conv1_grad_Opdef.add_input("conv1_w");
	conv1_grad_Opdef.add_input("conv1_grad");
	conv1_grad_Opdef.add_output("conv1_w_grad");
	conv1_grad_Opdef.add_output("conv1_b_grad");
	conv1_grad_Opdef.add_output("data_grad");
	conv1_grad_Opdef.set_engine("CUDNN");
	conv1_grad_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	AddArgument<int>("kernel", 5, &conv1_grad_Opdef);
	net_def.add_op()->CopyFrom(conv1_grad_Opdef);	
	/************************************************************/
	/*			  Add Gradient Update Operators 				*/
	/************************************************************/
	
	net_def.add_external_input("ONE");
	net_def.add_external_input("iter");
	// iter
	OperatorDef iter_Opdef;
	iter_Opdef.set_type("Iter");
	iter_Opdef.add_input("iter");
	iter_Opdef.add_output("iter");
	iter_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(iter_Opdef);	
	
	// LR
	OperatorDef LR_Opdef;
	LR_Opdef.set_type("LearningRate");
	LR_Opdef.add_input("iter");
	LR_Opdef.add_output("LR");
	AddArgument<std::string>("policy", "step", &LR_Opdef);
	AddArgument<int>("stepsize", 1, &LR_Opdef);
	AddArgument<float>("base_lr", -0.1, &LR_Opdef);
	AddArgument<float>("gamma", 0.999, &LR_Opdef);
	LR_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(LR_Opdef);	

	
	
	// conv1_w update
	OperatorDef conv1_w_update_Opdef;
	conv1_w_update_Opdef.set_type("WeightedSum");
	conv1_w_update_Opdef.add_input("conv1_w");
	conv1_w_update_Opdef.add_input("ONE");
	conv1_w_update_Opdef.add_input("conv1_w_grad");
	conv1_w_update_Opdef.add_input("LR");
	conv1_w_update_Opdef.add_output("conv1_w");
	conv1_w_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv1_w_update_Opdef);	
	
	// conv1_b update
	OperatorDef conv1_b_update_Opdef;
	conv1_b_update_Opdef.set_type("WeightedSum");
	conv1_b_update_Opdef.add_input("conv1_b");
	conv1_b_update_Opdef.add_input("ONE");
	conv1_b_update_Opdef.add_input("conv1_b_grad");
	conv1_b_update_Opdef.add_input("LR");
	conv1_b_update_Opdef.add_output("conv1_b");
	conv1_b_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv1_b_update_Opdef);	

	// conv2_w update
	OperatorDef conv2_w_update_Opdef;
	conv2_w_update_Opdef.set_type("WeightedSum");
	conv2_w_update_Opdef.add_input("conv2_w");
	conv2_w_update_Opdef.add_input("ONE");
	conv2_w_update_Opdef.add_input("conv2_w_grad");
	conv2_w_update_Opdef.add_input("LR");
	conv2_w_update_Opdef.add_output("conv2_w");
	conv2_w_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv2_w_update_Opdef);	
	
	// conv2_b update
	OperatorDef conv2_b_update_Opdef;
	conv2_b_update_Opdef.set_type("WeightedSum");
	conv2_b_update_Opdef.add_input("conv2_b");
	conv2_b_update_Opdef.add_input("ONE");
	conv2_b_update_Opdef.add_input("conv2_b_grad");
	conv2_b_update_Opdef.add_input("LR");
	conv2_b_update_Opdef.add_output("conv2_b");
	conv2_b_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(conv2_b_update_Opdef);
	
	// fc3_w update
	OperatorDef fc3_w_update_Opdef;
	fc3_w_update_Opdef.set_type("WeightedSum");
	fc3_w_update_Opdef.add_input("fc3_w");
	fc3_w_update_Opdef.add_input("ONE");
	fc3_w_update_Opdef.add_input("fc3_w_grad");
	fc3_w_update_Opdef.add_input("LR");
	fc3_w_update_Opdef.add_output("fc3_w");
	fc3_w_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(fc3_w_update_Opdef);
	
	// fc3_b update
	OperatorDef fc3_b_update_Opdef;
	fc3_b_update_Opdef.set_type("WeightedSum");
	fc3_b_update_Opdef.add_input("fc3_b");
	fc3_b_update_Opdef.add_input("ONE");
	fc3_b_update_Opdef.add_input("fc3_b_grad");
	fc3_b_update_Opdef.add_input("LR");
	fc3_b_update_Opdef.add_output("fc3_b");
	fc3_b_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(fc3_b_update_Opdef);

	// pred_w update
	OperatorDef pred_w_update_Opdef;
	pred_w_update_Opdef.set_type("WeightedSum");
	pred_w_update_Opdef.add_input("pred_w");
	pred_w_update_Opdef.add_input("ONE");
	pred_w_update_Opdef.add_input("pred_w_grad");
	pred_w_update_Opdef.add_input("LR");
	pred_w_update_Opdef.add_output("pred_w");
	pred_w_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(pred_w_update_Opdef);

	// pred_b update
	OperatorDef pred_b_update_Opdef;
	pred_b_update_Opdef.set_type("WeightedSum");
	pred_b_update_Opdef.add_input("pred_b");
	pred_b_update_Opdef.add_input("ONE");
	pred_b_update_Opdef.add_input("pred_b_grad");
	pred_b_update_Opdef.add_input("LR");
	pred_b_update_Opdef.add_output("pred_b");
	pred_b_update_Opdef.mutable_device_option()->set_device_type(DEVICE_TYPE);
	net_def.add_op()->CopyFrom(pred_b_update_Opdef);

}