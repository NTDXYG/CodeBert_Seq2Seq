from model import CodeBert_Seq2Seq

# 初始化模型
model = CodeBert_Seq2Seq(codebert_path = '/data/home/yangguang/models/codebert-base', decoder_layers = 6, fix_encoder = False, beam_size = 10,
                         max_source_length = 64, max_target_length = 64, load_model_path = None)

# 加载微调过的模型
# model = CodeBert_Seq2Seq(codebert_path = 'D:\\new_idea\\Final\\model\\codebert', decoder_layers = 6, fix_encoder = True, beam_size = 10,
#                          max_source_length = 256, max_target_length = 32, load_model_path = './valid_output/checkpoint-last/pytorch_model.bin')

# 模型训练
model.train(train_filename ='data/train.csv', train_batch_size = 64, num_train_epochs = 30, learning_rate = 5e-5,
            do_eval = True, dev_filename ='data/valid.csv', eval_batch_size = 64, output_dir ='valid_output')
#
# 模型测试
model.test(test_filename ='data/test.csv', test_batch_size = 16, output_dir ='test_output')

# 模型推理
comment = model.predict(source = 'public static boolean isStringType ( Type t ) { return t . equals ( RefType . v ( _STR ) ) ; }')
print(comment)