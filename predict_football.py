import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='足球比赛预测系统')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--predict', action='store_true', help='预测新比赛')
    parser.add_argument('--model-dir', default='models', help='模型保存目录')

    args = parser.parse_args()

    if not (args.train or args.predict):
        parser.print_help()
        return

    if args.train:
        print("开始训练模型...")
        from service.util.train_model import train_and_save_model
        train_and_save_model(args.model_dir)
        print("模型训练完成！")

    if args.predict:
        print("开始预测新比赛...")
        from service.util.predict_new_matches import predict_new_matches

        # 检查模型文件是否存在
        model_path = os.path.join(args.model_dir, 'best_model.pkl')
        model_path = os.path.join(args.model_dir, 'best_model.pkl')
        scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
        feature_names_path = os.path.join(args.model_dir, 'feature_names.pkl')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_names_path]):
            print(f"错误：在 {args.model_dir} 目录中未找到必要的模型文件")
            print("请先使用 --train 参数训练模型")
            return

        # 执行预测
        results = predict_new_matches(model_path, scaler_path, feature_names_path)

        if results is not None:
            print("\n预测完成！")

            # 显示高置信度预测
            high_conf = results[results.filter(like='prob_').max(axis=1) > 0.6]
            if not high_conf.empty:
                print("\n高置信度预测结果:")
                print(high_conf.to_string())

if __name__ == '__main__':
    main()
