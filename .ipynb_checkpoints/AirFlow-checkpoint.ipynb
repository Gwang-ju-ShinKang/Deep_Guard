{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "a8d7b9ba-c3f0-4128-b5f9-d556e3603ea6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pymysql in c:\\users\\smhrd\\anaconda3\\lib\\site-packages (1.1.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pymysql\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "35ea785f-b16e-441b-9e05-97b335dd6dea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2024-12-16 17:40:42.342158] ds_session_info 테이블 생성 완료\n",
      "[2024-12-16 17:40:42.450038] session_info 테이블 데이터 수: 154\n",
      "[2024-12-16 17:40:42.824884] ds_session_info 테이블 데이터 삭제 완료\n",
      "[2024-12-16 17:40:43.110636] session_info → ds_session_info 데이터 삽입 완료\n",
      "[2024-12-16 17:40:43.213847] ds_session_info 테이블 데이터 수: 154\n",
      "[2024-12-16 17:40:43.213847] ✅ 데이터 일치: 원천(154) == ODS(154)\n"
     ]
    }
   ],
   "source": [
    "import pymysql\n",
    "from datetime import datetime\n",
    "\n",
    "# MySQL 연결 설정\n",
    "def get_connection():\n",
    "    return pymysql.connect(\n",
    "        host=\"project-db-cgi.smhrd.com\",\n",
    "        user=\"cgi_24k_data_p3_3\",\n",
    "        password=\"smhrd3\",\n",
    "        database=\"cgi_24k_data_p3_3\",\n",
    "        port=3307,\n",
    "        charset=\"utf8mb4\"\n",
    "    )\n",
    "\n",
    "# 1. 원천 테이블 데이터 수 조회\n",
    "def get_table_count(table_name):\n",
    "    try:\n",
    "        conn = get_connection()\n",
    "        with conn.cursor() as cursor:\n",
    "            query = f\"SELECT COUNT(*) FROM {table_name}\"\n",
    "            cursor.execute(query)\n",
    "            result = cursor.fetchone()[0]\n",
    "        print(f\"[{datetime.now()}] {table_name} 테이블 데이터 수: {result}\")\n",
    "        return result\n",
    "    except Exception as e:\n",
    "        print(f\"오류 발생 (get_table_count): {e}\")\n",
    "    finally:\n",
    "        conn.close()\n",
    "\n",
    "# 2. ODS 테이블 데이터 삭제\n",
    "def delete_ods_table(table_name):\n",
    "    try:\n",
    "        conn = get_connection()\n",
    "        with conn.cursor() as cursor:\n",
    "            query = f\"DELETE FROM {table_name}\"\n",
    "            cursor.execute(query)\n",
    "        conn.commit()\n",
    "        print(f\"[{datetime.now()}] {table_name} 테이블 데이터 삭제 완료\")\n",
    "    except Exception as e:\n",
    "        print(f\"오류 발생 (delete_ods_table): {e}\")\n",
    "    finally:\n",
    "        conn.close()\n",
    "\n",
    "# 3. 원천 테이블 데이터를 ODS 테이블로 삽입\n",
    "def insert_into_ods(source_table, target_table):\n",
    "    try:\n",
    "        conn = get_connection()\n",
    "        with conn.cursor() as cursor:\n",
    "            query = f\"INSERT INTO {target_table} SELECT * FROM {source_table}\"\n",
    "            cursor.execute(query)\n",
    "        conn.commit()\n",
    "        print(f\"[{datetime.now()}] {source_table} → {target_table} 데이터 삽입 완료\")\n",
    "    except Exception as e:\n",
    "        print(f\"오류 발생 (insert_into_ods): {e}\")\n",
    "    finally:\n",
    "        conn.close()\n",
    "\n",
    "# 4. 원천과 ODS 테이블 데이터 수 비교\n",
    "def compare_counts(source_count, ods_count):\n",
    "    if source_count == ods_count:\n",
    "        print(f\"[{datetime.now()}] ✅ 데이터 일치: 원천({source_count}) == ODS({ods_count})\")\n",
    "    else:\n",
    "        print(f\"[{datetime.now()}] ❌ 데이터 불일치: 원천({source_count}) != ODS({ods_count})\")\n",
    "\n",
    "# 5. ODS 테이블 생성\n",
    "def create_ods_table_if_not_exists(source_table, target_table):\n",
    "    try:\n",
    "        conn = get_connection()\n",
    "        with conn.cursor() as cursor:\n",
    "            query = f\"CREATE TABLE IF NOT EXISTS {target_table} LIKE {source_table}\"\n",
    "            cursor.execute(query)\n",
    "        conn.commit()\n",
    "        print(f\"[{datetime.now()}] {target_table} 테이블 생성 완료\")\n",
    "    except Exception as e:\n",
    "        print(f\"오류 발생 (create_ods_table_if_not_exists): {e}\")\n",
    "    finally:\n",
    "        conn.close()\n",
    "\n",
    "# 메인 실행 함수\n",
    "def main():\n",
    "    source_table = \"session_info\"\n",
    "    ods_table = \"ds_session_info\"\n",
    "\n",
    "    # 1. ODS 테이블 존재 확인 및 생성\n",
    "    create_ods_table_if_not_exists(source_table, ods_table)\n",
    "\n",
    "    # 2. 원천 데이터 수 조회\n",
    "    source_count = get_table_count(source_table)\n",
    "\n",
    "    # 3. ODS 테이블 데이터 삭제\n",
    "    delete_ods_table(ods_table)\n",
    "\n",
    "    # 4. 원천 데이터를 ODS 테이블로 삽입\n",
    "    insert_into_ods(source_table, ods_table)\n",
    "\n",
    "    # 5. ODS 데이터 수 조회\n",
    "    ods_count = get_table_count(ods_table)\n",
    "\n",
    "    # 6. 데이터 수 비교\n",
    "    compare_counts(source_count, ods_count)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b71691e-e471-47e7-9998-ceabd134720e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4e89c9f-4421-4970-8bf4-03853db204ea",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "attachments": {
    "64c06c04-1ccf-4ee8-8e86-e6092916702e.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "id": "bae04495-7125-41d6-9913-50049a03dc2d",
   "metadata": {},
   "source": [
    "![image.png](attachment:64c06c04-1ccf-4ee8-8e86-e6092916702e.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f3848cf-ec4e-46e6-85db-ab1bb12885bd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cb9b6ac-057c-4c64-940a-417afb6a7ddb",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9e7c18d-7c9f-4b3d-bc64-6b9566962816",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9b3c45f-c6dd-4a25-ad97-29200b978f87",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a58488d-8e88-40d7-9cdf-bd83e32ffc82",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "595d7111-40eb-438a-969d-bc70ac1f7ffa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71332ccb-aee8-4bc2-8a77-7ba56bed0fe9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "91278cf1-58ad-45fd-bebd-ff688e2bf500",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abdefe1c-e87a-4763-97de-16e4645e59a4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36b345a4-953f-4a3b-8d08-b270e4b6f695",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a65c2c84-e6c5-40b7-a412-3f3213edf51a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7df80512-6c3d-46f6-b639-ff81fb1c140e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0a8d924-180c-4aa7-893c-c3127eeece33",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dab788ff-4a66-436c-9331-1128919ce586",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
