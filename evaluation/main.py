from utils import *

def main() -> None:
    os.environ['PYTHONHASHSEED'] = '0'
    logging.basicConfig(level=logging.DEBUG)
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--base-url', type=str, default=None)
    parser.add_argument('--openai-api-key', type=str, nargs='+')
    parser.add_argument('--openai-api-key-file', type=Path, default=None)
    parser.add_argument('--ans1', type=str, default="original_answer")
    parser.add_argument('--ans2', type=str, default="correction_answer")
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=DEFAULT_OPENAI_CHAT_COMPLETION_MODELS,
    )
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('./cache'))
    parser.add_argument('--num-cpus', type=int, default=max(os.cpu_count() - 4, 4))
    parser.add_argument('--num-workers', type=int, default=max(2 * (os.cpu_count() - 4) // 3, 4))
    parser.add_argument('--type', type=str, default="")
    parser.add_argument('--platform',  type=str, default="openai")
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.num_workers >= args.num_cpus:
        raise ValueError('num_workers should be less than num_cpus')

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    cache_dir = args.cache_dir.expanduser().absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_gitignore_file = cache_dir / '.gitignore'
    if not cache_gitignore_file.exists():
        with cache_gitignore_file.open(mode='wt', encoding='utf-8') as f:
            f.write('*\n')

    input_file = args.input_file.expanduser().absolute()
    output_file = args.output_dir.expanduser().absolute()
    inputs = prepare_inputs(input_file, shuffle=args.shuffle, platform=args.platform, type=args.type)

    openai_api_keys = get_openai_api_keys(args.openai_api_key, args.openai_api_key_file)

    print(len(openai_api_keys), args.num_cpus)

    print(openai_api_keys)
    ray.init()

    results = batch_request_openai(
        inputs=inputs,
        openai_api_keys=openai_api_keys,
        openai_models=args.models,
        base_url=args.base_url,
        num_workers=args.num_workers,
        cache_dir=cache_dir,
    )
    processed_results = []
    raw_results = []
    for result in results:
        raw_results.append(result)
        new_result = post_process(result, args.type)
        processed_results.append(new_result)

    # 这一点非常重要，确保所有的结果都已经被收集
    assert all(result is not None for result in results)

    ray.shutdown()

    output_file.write_text(json.dumps(processed_results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
