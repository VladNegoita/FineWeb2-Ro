import pyarrow.parquet as pq
import pyarrow as pa
import ijson
from tqdm import tqdm
from argparse import ArgumentParser


def json_to_parquet_stream(input_json, output_parquet, batch_size=500, total_records=None):
    print(f"Converting {input_json} to {output_parquet} with batch size {batch_size}")

    schema = pa.schema([
        pa.field("text", pa.string()),
    ])
    
    batch = []
    writer = pq.ParquetWriter(output_parquet, schema)
    with open(input_json, 'rb') as f:
        parser = ijson.items(f, 'item')
        for record in tqdm(parser, total=total_records, unit='rec'):
            batch.append({"text": record["text"]})
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch)
                writer.write_table(table)
                batch.clear()

        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(output_parquet, table.schema)
            writer.write_table(table)

    if writer:
        writer.close()

    print(f"Conversion complete. Parquet file saved to {output_parquet}", flush=True)


def main():
    parser = ArgumentParser(description="Convert JSON to Parquet")
    parser.add_argument("--input_file", type=str, help="Input JSON file path", required=True)
    parser.add_argument("--output_file", type=str, help="Output Parquet file path", required=True)
    
    args = parser.parse_args()
    json_to_parquet_stream(args.input_file, args.output_file)


if __name__ == "__main__":
    main()