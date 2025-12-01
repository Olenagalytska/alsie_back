import requests
import json
import sys
import time
from typing import Optional

BASE_URL = "http://localhost:8000"


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subsection(title: str):
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)


def test_server_running():
    print("\n🔌 Перевірка чи запущений сервер...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Сервер працює!")
            print(f"   Версія: {data.get('version')}")
            print(f"   Повідомлення: {data.get('message')}")
            return True
        else:
            print(f"❌ Сервер повернув код {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Сервер не відповідає: {e}")
        print(f"\n💡 Запустіть сервер командою:")
        print("   python main.py")
        return False


def test_health():
    print_subsection("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data.get('status')}")
            print(f"   Xano налаштовано: {data.get('xano_configured')}")
            print(f"   OpenAI налаштовано: {data.get('openai_configured')}")
            
            if not data.get('xano_configured'):
                print("\n⚠️  УВАГА: Xano API не налаштовано!")
                print("   Додайте XANO_API_KEY в .env файл")
            
            if not data.get('openai_configured'):
                print("\n⚠️  УВАГА: OpenAI API не налаштовано!")
                print("   Додайте OPENAI_API_KEY в .env файл")
            
            return data.get('xano_configured') and data.get('openai_configured')
        else:
            print(f"❌ Помилка {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False


def test_chat_history(ub_id: int):
    print_subsection(f"Історія чату для UB ID: {ub_id}")
    try:
        response = requests.get(f"{BASE_URL}/chat/{ub_id}/history")
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            count = data.get('count', 0)
            messages = data.get('messages', [])
            
            print(f"✅ Знайдено повідомлень: {count}")
            
            if count > 0:
                print(f"\n📝 Останні 5 повідомлень:")
                for i, msg in enumerate(messages[-5:], 1):
                    role = msg.get('role', 'unknown')
                    msg_id = msg.get('id', 'N/A')
                    prev_id = msg.get('prev_id', 'N/A')
                    
                    if role == 'user':
                        content = msg.get('user_content', {}).get('text', '')
                    else:
                        ai_content = msg.get('ai_content', [])
                        content = ai_content[0].get('text', '') if ai_content else ''
                    
                    print(f"\n  {i}. [{role.upper()}] (ID: {msg_id}, prev: {prev_id})")
                    print(f"     {content[:100]}{'...' if len(content) > 100 else ''}")
            
            return True
        else:
            print(f"❌ Помилка {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False


def test_send_message(ub_id: int, message: str):
    print_subsection(f"Відправка повідомлення до UB ID: {ub_id}")
    print(f"📤 Повідомлення: {message}")
    
    try:
        print("\n⏳ Відправляємо запит до API...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/message",
            json={
                "ub_id": ub_id,
                "content": message
            },
            timeout=120
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Час відповіді: {elapsed_time:.2f}s")
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Відповідь отримано:")
            print(f"   Type: {data.get('type')}")
            print(f"   Title: {data.get('title')}")
            print(f"\n📥 Текст відповіді AI:")
            print(f"   {data.get('text')}")
            
            if data.get('additional'):
                print(f"\n📋 Додаткова інформація:")
                print(f"   {json.dumps(data.get('additional'), indent=2, ensure_ascii=False)}")
            
            return True
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"Відповідь: {response.text}")
            return False
    except requests.Timeout:
        print(f"❌ Таймаут запиту (>120s)")
        return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False


def test_evaluation(ub_id: int):
    print_subsection(f"Оцінювання чату UB ID: {ub_id}")
    
    try:
        print("\n⏳ Запускаємо оцінювання...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/{ub_id}/evaluate",
            timeout=120
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Час оцінювання: {elapsed_time:.2f}s")
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Оцінювання завершено:")
            print(f"\n📊 Результат:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"Відповідь: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False


def test_clear_memory(ub_id: int):
    print_subsection(f"Очищення пам'яті для UB ID: {ub_id}")
    
    try:
        response = requests.post(f"{BASE_URL}/chat/{ub_id}/clear-memory")
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {data.get('status')}")
            return True
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"Відповідь: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False


def interactive_mode():
    print_section("🎮 Інтерактивний режим тестування")
    
    print("\n📋 Що ви хочете протестувати?")
    print("1. Історію чату (потрібен UB ID)")
    print("2. Відправити повідомлення (потрібен UB ID)")
    print("3. Запустити оцінювання (потрібен UB ID)")
    print("4. Очистити пам'ять агента (потрібен UB ID)")
    print("5. Повний тест (всі функції)")
    
    choice = input("\nВаш вибір (1-5): ").strip()
    
    if choice in ["1", "2", "3", "4", "5"]:
        ub_id_input = input("\nВведіть UB ID (наприклад, 12518): ").strip()
        
        try:
            ub_id = int(ub_id_input)
        except ValueError:
            print("❌ Невірний UB ID")
            return
        
        if choice == "1":
            test_chat_history(ub_id)
        
        elif choice == "2":
            message = input("\nВведіть повідомлення для студента: ").strip()
            if message:
                test_send_message(ub_id, message)
                
                if input("\nПоказати оновлену історію? (y/n): ").strip().lower() == 'y':
                    test_chat_history(ub_id)
        
        elif choice == "3":
            test_evaluation(ub_id)
        
        elif choice == "4":
            test_clear_memory(ub_id)
        
        elif choice == "5":
            print_section("Повний тест")
            
            print("\n1️⃣ Історія до відправки повідомлення:")
            test_chat_history(ub_id)
            
            message = input("\nВведіть тестове повідомлення: ").strip()
            if message:
                print("\n2️⃣ Відправка повідомлення:")
                if test_send_message(ub_id, message):
                    
                    print("\n3️⃣ Історія після відправки:")
                    test_chat_history(ub_id)
                    
                    if input("\nЗапустити оцінювання? (y/n): ").strip().lower() == 'y':
                        print("\n4️⃣ Оцінювання:")
                        test_evaluation(ub_id)
                    
                    if input("\nОчистити пам'ять? (y/n): ").strip().lower() == 'y':
                        print("\n5️⃣ Очищення пам'яті:")
                        test_clear_memory(ub_id)
    else:
        print("❌ Невірний вибір")


def quick_test(ub_id: int, message: Optional[str] = None):
    print_section(f"⚡ Швидкий тест для UB ID: {ub_id}")
    
    print("\n1️⃣ Перевірка історії:")
    test_chat_history(ub_id)
    
    if message:
        print("\n2️⃣ Відправка тестового повідомлення:")
        test_send_message(ub_id, message)
        
        print("\n3️⃣ Оновлена історія:")
        test_chat_history(ub_id)


def main():
    print_section("EdTech AI Platform - Тестування OpenAI Agents")
    
    if not test_server_running():
        sys.exit(1)
    
    print_section("Базові перевірки")
    
    if not test_health():
        print("\n⚠️  Конфігурація неповна. Перевірте .env файл")
        if input("\nПродовжити тестування? (y/n): ").strip().lower() != 'y':
            sys.exit(1)
    
    if len(sys.argv) > 1:
        try:
            ub_id = int(sys.argv[1])
            message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
            quick_test(ub_id, message)
        except ValueError:
            print("❌ Невірний UB ID")
            sys.exit(1)
    else:
        interactive_mode()
    
    print_section("✅ Тестування завершено!")
    print("\n💡 Підказки:")
    print("  - Для швидкого тесту: python test_agents.py <UB_ID>")
    print("  - З повідомленням: python test_agents.py <UB_ID> 'Ваше повідомлення'")
    print("  - Інтерактивний режим: python test_agents.py")


if __name__ == "__main__":
    main()