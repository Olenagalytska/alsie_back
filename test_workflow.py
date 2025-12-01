import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"


def check_server():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Сервер не відповідає")
            return False
        print("✅ Сервер запущений\n")
        return True
    except:
        print("❌ Сервер не запущений. Запустіть: python main.py")
        return False


def show_state(ub_id: int):
    try:
        response = requests.get(f"{BASE_URL}/chat/{ub_id}/state")
        if response.status_code == 200:
            state = response.json()
            total_q = len(state.get('questions', []))
            current = state.get('current_question_index', 0)
            
            print(f"📊 Стан workflow:")
            print(f"   Питання: {current + 1}/{total_q}")
            print(f"   Follow-ups: {state.get('follow_up_count', 0)}/{state.get('max_follow_ups', 3)}")
            print(f"   Статус: {state.get('status', 'unknown')}")
            
            answers = state.get('answers', [])
            if answers:
                print(f"   Відповідей: {len(answers)}")
                last = answers[-1]
                if last.get('evaluation'):
                    ev = last['evaluation']
                    print(f"   Остання оцінка:")
                    print(f"     - Повна: {ev.get('complete', False)}")
                    if ev.get('missing_concepts'):
                        print(f"     - Не вистачає: {', '.join(ev['missing_concepts'])}")
            print()
            return state
        else:
            print("⚠️  Стан не знайдено (новий чат)\n")
            return None
    except Exception as e:
        print(f"⚠️  Помилка завантаження стану: {e}\n")
        return None


def send_message(ub_id: int, message: str, step_name: str = ""):
    if step_name:
        print(f"{'─'*70}")
        print(f"{step_name}")
        print(f"{'─'*70}")
    
    print(f"📤 Відправка: '{message}'\n")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/message",
            json={"ub_id": ub_id, "content": message},
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Час відповіді: {elapsed:.2f}s")
        print(f"📊 Статус: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            ai_text = data.get('text', '')
            
            print(f"✅ Відповідь AI:")
            print(f"{ai_text}\n")
            
            show_state(ub_id)
            return True
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"{response.text[:300]}\n")
            return False
            
    except requests.Timeout:
        print("❌ Таймаут (>120s)\n")
        return False
    except Exception as e:
        print(f"❌ Помилка: {e}\n")
        return False


def test_scenarios(ub_id: int):
    print(f"{'='*70}")
    print(f" Тестування Workflow для UB ID: {ub_id}")
    print(f"{'='*70}\n")
    
    if not check_server():
        return False
    
    show_state(ub_id)
    
    scenarios = [
        {
            "name": "1. Неповна відповідь",
            "message": "Ембединг",
            "expect": "Має отримати follow-up питання"
        },
        {
            "name": "2. Неправильна відповідь",
            "message": "Не знаю",
            "expect": "Може отримати follow-up або перейти далі"
        },
        {
            "name": "3. Повна правильна відповідь",
            "message": "Позиційне кодування додає інформацію про послідовність слів у реченні",
            "expect": "Має перейти до наступного питання"
        }
    ]
    
    print("Оберіть сценарій:\n")
    for i, sc in enumerate(scenarios, 1):
        print(f"{i}. {sc['name']}")
        print(f"   Повідомлення: '{sc['message']}'")
        print(f"   Очікується: {sc['expect']}\n")
    
    print("4. Власне повідомлення")
    print("5. Запустити всі сценарії послідовно\n")
    
    choice = input("Ваш вибір (1-5): ").strip()
    
    if choice in ["1", "2", "3"]:
        sc = scenarios[int(choice) - 1]
        send_message(ub_id, sc["message"], sc["name"])
    
    elif choice == "4":
        msg = input("\nВведіть повідомлення: ").strip()
        if msg:
            send_message(ub_id, msg, "Власне повідомлення")
    
    elif choice == "5":
        print("\n🚀 Запуск всіх сценаріїв...\n")
        for sc in scenarios:
            send_message(ub_id, sc["message"], sc["name"])
            time.sleep(1)
    else:
        print("❌ Невірний вибір")
        return False
    
    print(f"{'='*70}")
    print("📊 Фінальний стан")
    print(f"{'='*70}\n")
    
    show_state(ub_id)
    return True


def show_info():
    print(f"\n{'='*70}")
    print(" ℹ️  Інформація про Workflow")
    print(f"{'='*70}\n")
    
    print("🔄 Логіка роботи:\n")
    print("1. Interviewer Agent ставить питання")
    print("2. Студент відповідає")
    print("3. Evaluator Agent оцінює відповідь:")
    print("   • complete: true → наступне питання")
    print("   • complete: false + follow_up_count < 3 → follow-up")
    print("   • follow_up_count >= 3 → наступне питання")
    print("4. Стан зберігається в Xano workflow_state\n")
    
    print("📋 Структура WorkflowState:")
    print("   • questions[] - список питань з key_concepts")
    print("   • current_question_index - поточне питання")
    print("   • answers[] - відповіді з оцінками")
    print("   • follow_up_count - лічильник уточнень")
    print("   • status - active/finished\n")
    
    print("🎯 Agents SDK:")
    print("   • Interviewer - задає питання, не підказує")
    print("   • Evaluator - перевіряє наявність key_concepts\n")
    
    print("🔍 Трейсинг:")
    print("   https://platform.openai.com/traces\n")


def main():
    print("\n🎓 EdTech AI Platform - Workflow Test\n")
    
    if len(sys.argv) < 2:
        print("Використання:")
        print("  python test_workflow.py <UB_ID>")
        print("  python test_workflow.py info\n")
        print("Приклад:")
        print("  python test_workflow.py 12610")
        sys.exit(1)
    
    if sys.argv[1] == "info":
        show_info()
        sys.exit(0)
    
    try:
        ub_id = int(sys.argv[1])
    except ValueError:
        print("❌ UB_ID має бути числом")
        sys.exit(1)
    
    success = test_scenarios(ub_id)
    
    if success:
        print("\n✅ Тест завершено!")
        print("\n💡 Корисні команди:")
        print(f"  curl http://localhost:8000/chat/{ub_id}/state")
        print(f"  python test_workflow.py {ub_id}")
    else:
        print("\n❌ Тест завершився з помилками")
        sys.exit(1)


if __name__ == "__main__":
    main()